"""Live viewing preserves exact images without exposing transport state."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any, cast

from google.genai import types

from langslice.linear.live import LiveEvents


def event(parts, *, partial=False):
    return NS(content=NS(parts=parts), partial=partial)


def test_live_seed_and_tool_images_preserve_bytes_and_hide_signatures():
    records = []
    live = LiveEvents(records.append)
    image = types.Part.from_bytes(data=b'exact image bytes', mime_type='image/png')
    live.start(NS(model='test-model', instruction='job'), NS(parts=[
        types.Part.from_text(text='Section 1'), image,
    ]))
    live.event(event([NS(text=None, function_call=None, function_response=NS(
        name='compare', id='c1', response={'ok': True, 'encrypted_content': 'secret'},
        parts=[image],
    ))]))
    assert records[1]['images'] == [
        {'data': b'exact image bytes', 'mime_type': 'image/png', 'label': 'Section 1'}
    ]
    assert records[2]['images'][0]['data'] == b'exact image bytes'
    assert records[2]['response'] == {'ok': True}
    assert 'secret' not in str(records)


def test_streaming_final_text_and_calls_are_not_duplicated():
    records = []
    live = LiveEvents(records.append)
    for text in ['Hel', 'lo']:
        live.event(event([types.Part.from_text(text=text)], partial=True))
    live.event(event([types.Part(thought=True, text='Looking', thought_signature=b'secret')],
                     partial=True))
    call = types.Part(function_call=types.FunctionCall(name='look', id='c1', args={}))
    live.event(event([call], partial=True))
    live.event(event([types.Part.from_text(text='Hello!'),
                      types.Part(thought=True, text='Looking now', thought_signature=b'secret'),
                      call]))
    assert ''.join(r['text'] for r in records if r['kind'] == 'text') == 'Hello!'
    assert ''.join(r['text'] for r in records if r['kind'] == 'reasoning') == 'Looking now'
    assert len([r for r in records if r['kind'] == 'tool_call']) == 1
    assert 'secret' not in str(records)
    live.event(event([types.Part.from_text(text='Next response')]))
    assert records[-1] == {'kind': 'text', 'text': 'Next response'}


def test_observer_failures_do_not_fail_the_agent(caplog):
    def broken(record):
        raise RuntimeError('viewer closed')
    LiveEvents(broken).emit('text', text='hello')
    assert 'Live session observer failed' in caplog.text


def test_final_revision_is_visible_when_it_differs_from_streamed_text():
    records = []
    live = LiveEvents(records.append)
    live.event(event([
        types.Part.from_text(text='Initial answer'),
        types.Part(thought=True, text='Initial summary'),
    ], partial=True))
    live.event(event([
        types.Part.from_text(text='Revised answer'),
        types.Part(thought=True, text='Revised summary'),
    ]))
    assert records[-2:] == [
        {'kind': 'text', 'text': 'Revised answer', 'revised': True},
        {'kind': 'reasoning', 'text': 'Revised summary', 'revised': True},
    ]
    live.event(event([types.Part.from_text(text='Next answer')]))
    assert records[-1] == {'kind': 'text', 'text': 'Next answer'}


def test_session_streaming_counts_final_calls_once_and_debrief_once(monkeypatch):
    from langslice.linear import session
    records = []
    options = []
    submitted = [False]
    call = types.Part(function_call=types.FunctionCall(name='submit', id='c1', args={}))

    def wrapped(parts, partial=False):
        value = event(parts, partial=partial)
        value.get_function_calls = lambda: [p.function_call for p in parts if p.function_call]
        value.get_function_responses = lambda: []
        return value

    class Runner:
        def __init__(self, **kwargs):
            self.session_service = self
        async def create_session(self, **kwargs):
            pass
        async def run_async(self, **kwargs):
            options.append(kwargs)
            if len(options) == 1:
                yield wrapped([call], partial=True)
                yield wrapped([call])
                submitted[0] = True
                yield wrapped([])
            else:
                yield wrapped([types.Part.from_text(text='Deb')], partial=True)
                yield wrapped([types.Part.from_text(text='rief')], partial=True)
                yield wrapped([types.Part.from_text(text='Debrief')])

    monkeypatch.setattr(session, 'InMemoryRunner', Runner)
    monkeypatch.setattr(session, 'App', lambda **kwargs: kwargs)
    monkeypatch.setattr(session, 'build_plugins', lambda *args, **kwargs: [])
    monkeypatch.setattr(session, 'open_trace', lambda *args, **kwargs: None)
    sink = []
    result = asyncio.run(session.run_agent_session(
        agent=cast(Any, NS(model='test', instruction='job')),
        seed_message=types.Content(role='user', parts=[types.Part.from_text(text='seed')]),
        done=lambda: submitted[0], nudge_no_tool='continue', nudge_continue='continue',
        max_iterations=2, run_label='test', debrief='debrief', debrief_sink=sink,
        on_event=records.append,
    ))
    assert result == (1, 1)
    assert sink == ['Debrief']
    assert all(v['run_config'].streaming_mode == session.StreamingMode.SSE for v in options)
    assert records[-1]['kind'] == 'complete'
    assert records[-1]['submitted'] is True


def test_session_closes_runner_in_original_context_before_debrief(monkeypatch):
    from contextvars import ContextVar

    from langslice.linear import session

    context = ContextVar('test_runner_context', default='outside')
    closed = []
    submitted = [False]
    records = []

    class Runner:
        def __init__(self, **kwargs):
            self.session_service = self
            self.calls = 0
        async def create_session(self, **kwargs):
            pass
        async def run_async(self, **kwargs):
            self.calls += 1
            index = self.calls
            assert context.get() == 'outside', 'previous invocation was not closed'
            token = context.set(f'invocation-{index}')
            try:
                if index == 1:
                    submitted[0] = True
                value = event([types.Part.from_text(text='done')])
                value.get_function_calls = lambda: []
                value.get_function_responses = lambda: []
                yield value
                if index == 1:
                    raise AssertionError('submit must stop before another model turn')
            finally:
                # Like OpenTelemetry: reset must happen in the creator context.
                context.reset(token)
                closed.append(index)

    monkeypatch.setattr(session, 'InMemoryRunner', Runner)
    monkeypatch.setattr(session, 'App', lambda **kwargs: kwargs)
    monkeypatch.setattr(session, 'build_plugins', lambda *args, **kwargs: [])
    monkeypatch.setattr(session, 'open_trace', lambda *args, **kwargs: None)

    def observe(record):
        if record['kind'] == 'complete':
            assert closed == [1, 2]
            assert context.get() == 'outside'
        records.append(record)

    async def run():
        await session.run_agent_session(
            agent=cast(Any, NS(model='test', instruction='job')),
            seed_message=types.Content(role='user', parts=[]),
            done=lambda: submitted[0], nudge_no_tool='continue', nudge_continue='continue',
            max_iterations=2, run_label='test-close', debrief='debrief',
            on_event=observe,
        )
        assert closed == [1, 2], 'runner cleanup was deferred past session return'
        assert context.get() == 'outside'

    asyncio.run(run())
    assert records[-1]['kind'] == 'complete'
