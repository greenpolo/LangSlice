"""Landmark coordinates remain attached to anatomy through crop, affine and TPS."""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
from PIL import Image
from test_linear_physical import _ctx

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.landmark_warp import fit_spline
from langslice.linear.landmark_tools import image_to_section, section_to_image
from langslice.linear.toolbox import build_tools, damaged_transform_error

RAW = np.array([[100, 100], [400, 100], [100, 400], [400, 400], [250, 250]], float)


@pytest.fixture
def env(tmp_path):
    y, x = np.mgrid[:512, :512]
    rgb = np.stack((x // 2, y // 2, (x + y) // 4), axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / 's.tif', dpi=(2540, 2540))
    ctx, state = _ctx(tmp_path)
    state.slices[0].position_mm = .2
    box = build_tools(state, ctx, ctx.spec)
    return {fn.__name__: fn for fn in box.tools}, box, state, ctx


def pairs_for(view, source=RAW, target=None):
    target = source if target is None else target
    a = section_to_image(source, view['images_info'][0])
    b = section_to_image(target, view['images_info'][1])
    return [{'slice_xy': x.tolist(), 'atlas_xy': y.tolist(), 'label': f'point {i}'}
            for i, (x, y) in enumerate(zip(a, b, strict=True))]


def pixels(part):
    return np.asarray(Image.open(io.BytesIO(part.inline_data.data)).convert('RGB'))


def test_pixel_frames_match_images_caption_and_crop(env):
    tools, _, _, _ = env
    whole = tools['view_landmarks']('s.tif')
    cropped = tools['view_landmarks']('s.tif', zoom=[.2, .15, .8, .85])
    assert whole['status'] == cropped['status'] == 'ok'
    for result in [whole, cropped]:
        for frame, part in zip(result['images_info'], result[TOOL_MEDIA_PARTS_KEY], strict=True):
            image = pixels(part)
            assert image.shape[:2] == (frame['height'], frame['width'])
            assert frame['content_box'][1] > 0
            center = np.array([[frame['width'] / 2, sum(frame['content_box'][1::2]) / 2]])
            mapped = image_to_section(center, frame)
            np.testing.assert_allclose(section_to_image(mapped, frame), center, atol=1e-10)
            with pytest.raises(ValueError, match='caption'):
                image_to_section(np.array([[2, 0]]), frame)
    assert cropped['images_info'][0]['width'] < whole['images_info'][0]['width']
    # Independent pixel-center calculation checks the conversion, not just its inverse.
    frame = cropped['images_info'][0]
    canvas = np.array(frame['canvas_box'])
    content = np.array(frame['content_box'])
    expected = canvas[:2] + .5 * (canvas[2:] - canvas[:2]) / (content[2:] - content[:2])
    expected -= .5 + np.array(frame['section_offset'])
    np.testing.assert_allclose(image_to_section(content[None, :2], frame)[0], expected)


def test_warp_is_checkpointed_undoable_and_nonidentity_damage_accepted(env):
    tools, box, state, ctx = env
    state.slices[0].damaged = True
    before = copy.deepcopy(state.to_dict())
    view = tools['view_landmarks']('s.tif')
    target = RAW.copy()
    target[-1] += [14, -8]
    result = tools['warp_landmarks']('s.tif', view['view_id'], pairs_for(view, target=target),
                                     method='spline')
    assert result['status'] == 'ok', result
    assert max(result['landmark_residuals_mm']) < 2e-6
    assert len(box.undo_stack) == 1
    transform = state.slices[0].transform
    np.testing.assert_allclose(transform['spline']['source'], RAW / 512, atol=1e-10)
    np.testing.assert_allclose(transform['spline']['target'], target / 512, atol=1e-10)
    assert damaged_transform_error(state, ctx.spec) is None
    after = copy.deepcopy(state.to_dict())
    assert json.loads(open(ctx.checkpoint_path).read()) == after
    assert tools['undo']()['status'] == 'ok'
    assert state.to_dict() == before
    assert tools['redo']()['status'] == 'ok'
    assert state.to_dict() == after
    stale = tools['warp_landmarks']('s.tif', view['view_id'], pairs_for(view), method='spline')
    assert stale['error'] == 'STALE_LANDMARK_VIEW'


def test_affine_then_tps_then_tps_preserves_original_source_frame(env):
    tools, _, state, _ = env
    assert tools['adjust_transform']('s.tif', 3, 1.02, .98, .08, -.05)['status'] == 'ok'
    shown = RAW.copy()
    first_view = tools['view_landmarks']('s.tif')
    target = shown.copy()
    target[-1] += [12, -9]
    first = tools['warp_landmarks']('s.tif', first_view['view_id'],
                                    pairs_for(first_view, shown, target), method='spline')
    assert first['status'] == 'ok', first
    np.testing.assert_allclose(state.slices[0].transform['spline']['source'], RAW / 512, atol=1e-10)
    second_view = tools['view_landmarks']('s.tif')
    current = fit_spline(state.slices[0].transform['spline']).forward(RAW * .01) / .01
    np.testing.assert_allclose(current, target, atol=.0002)
    for index, pair in enumerate(second_view['existing_pairs']):
        recovered = image_to_section(np.array([pair['slice_xy']]), second_view['images_info'][0])
        np.testing.assert_allclose(recovered[0], RAW[index], atol=.0002)
    second_target = target.copy()
    second_target[-1] += [-5, 4]
    second = tools['warp_landmarks']('s.tif', second_view['view_id'],
                                     pairs_for(second_view, RAW, second_target), method='spline')
    assert second['status'] == 'ok', second
    spline = state.slices[0].transform['spline']
    np.testing.assert_allclose(spline['source'], RAW / 512, atol=1e-7)
    np.testing.assert_allclose(spline['target'], second_target / 512, atol=1e-10)
    # The prior result and the next before picture show the same full transform.
    # JPEG encoding + captions can contaminate their first block; compare anatomy below it.
    last_after = pixels(first[TOOL_MEDIA_PARTS_KEY][1])
    next_before = pixels(second[TOOL_MEDIA_PARTS_KEY][0])
    np.testing.assert_array_equal(last_after[40:], next_before[40:])
    assert np.abs(pixels(second[TOOL_MEDIA_PARTS_KEY][1])[40:].astype(float)
                  - next_before[40:]).max() > 5


@pytest.mark.parametrize('failure', ['fold', 'caption', 'outside_source', 'too_few'])
def test_refused_warps_leave_state_and_history_untouched(env, failure):
    tools, box, state, _ = env
    view = tools['view_landmarks']('s.tif')
    target = RAW.copy()
    if failure == 'fold':
        target[[1, 2]] = target[[2, 1]]
    pairs = pairs_for(view, target=target)
    if failure == 'caption':
        pairs[0]['slice_xy'] = [20, 0]
    elif failure == 'outside_source':
        pairs[0]['slice_xy'] = [-30, 50]
    elif failure == 'too_few':
        pairs = pairs[:3]
    before = copy.deepcopy(state.to_dict())
    result = tools['warp_landmarks']('s.tif', view['view_id'], pairs, method='spline')
    assert result['status'] == 'refused', result
    assert state.to_dict() == before
    assert not box.undo_stack


def test_crop_coordinates_and_stale_position(env):
    tools, _, state, _ = env
    view = tools['view_landmarks']('s.tif', zoom=[.1, .1, .9, .9])
    target = RAW.copy()
    target[-1] += [8, 6]
    result = tools['warp_landmarks']('s.tif', view['view_id'], pairs_for(view, target=target),
                                     method='spline')
    assert result['status'] == 'ok', result
    np.testing.assert_allclose(state.slices[0].transform['spline']['source'], RAW / 512, atol=1e-10)
    new_view = tools['view_landmarks']('s.tif')
    state.slices[0].position_mm += .025
    stale = tools['warp_landmarks']('s.tif', new_view['view_id'], pairs_for(new_view),
                                    method='spline')
    assert stale['error'] == 'STALE_LANDMARK_VIEW'


def test_landmark_writes_follow_interactive_task_permission(env):
    _, _, state, ctx = env
    ctx.spec.transform.interactive = False
    names = {fn.__name__ for fn in build_tools(state, ctx, ctx.spec).tools}
    assert 'view_landmarks' not in names
    assert 'warp_landmarks' not in names


def test_identity_spline_does_not_satisfy_damaged_correction(env):
    tools, _, state, ctx = env
    state.slices[0].damaged = True
    view = tools['view_landmarks']('s.tif')
    result = tools['warp_landmarks']('s.tif', view['view_id'], pairs_for(view), method='spline')
    assert result['status'] == 'ok', result
    failure = damaged_transform_error(state, ctx.spec)
    assert failure['error'] == 'DAMAGED_REQUIRES_MANUAL_TRANSFORM'


def test_edit_apply_keeps_raw_slice_fixed_and_edits_only_named_pair(env):
    tools, _, state, _ = env
    view = tools['view_landmarks']('s.tif')
    target = RAW.copy()
    target[-1] += [16, -9]
    edit = tools['edit_landmarks']('s.tif', view['view_id'],
                                    add=pairs_for(view, target=target))
    assert edit['status'] == 'ok', edit
    assert state.slices[0].transform is None
    draft = copy.deepcopy(state.slices[0].landmark_pairs)
    marked = tools['view_landmarks']('s.tif')
    result = tools['warp_landmarks']('s.tif', marked['view_id'], method='spline')
    assert result['status'] == 'ok', result
    after = tools['view_landmarks']('s.tif')
    # Applying a warp must not move either point-editing image or its numbered markers.
    for index in [0, 1]:
        np.testing.assert_array_equal(pixels(marked[TOOL_MEDIA_PARTS_KEY][index]),
                                      pixels(after[TOOL_MEDIA_PARTS_KEY][index]))
    chosen = after['existing_pairs'][-1]
    modified_target = np.asarray(chosen['atlas_xy']) + [2, -1]
    changed = tools['edit_landmarks']('s.tif', after['view_id'],
                                     move=[{'id': chosen['id'],
                                            'atlas_xy': modified_target.tolist()}])
    assert changed['status'] == 'ok', changed
    updated = state.slices[0].landmark_pairs
    assert updated[:-1] == draft[:-1]
    assert updated[-1]['source'] == draft[-1]['source']
    assert updated[-1]['target'] != draft[-1]['target']
    # Editing the draft must leave the applied transform intact until explicit application.
    np.testing.assert_allclose(state.slices[0].transform['spline']['target'], target / 512)


def test_pair_ids_checkpoint_resume_delete_and_undo(env):
    tools, _, state, ctx = env
    view = tools['view_landmarks']('s.tif')
    edit = tools['edit_landmarks']('s.tif', view['view_id'], add=pairs_for(view))
    assert edit['status'] == 'ok', edit
    first = copy.deepcopy(state.to_dict())
    ids = [p['id'] for p in state.slices[0].landmark_pairs]
    tools['undo']()
    assert not state.slices[0].landmark_pairs
    tools['redo']()
    assert state.to_dict() == first
    from langslice.linear.state import StackState
    resumed = StackState.from_dict(json.loads(open(ctx.checkpoint_path).read()))
    resumed_tools = {fn.__name__: fn for fn in build_tools(resumed, ctx, ctx.spec).tools}
    resumed_view = resumed_tools['view_landmarks']('s.tif')
    assert [p['id'] for p in resumed_view['existing_pairs']] == ids
    result = resumed_tools['edit_landmarks']('s.tif', resumed_view['view_id'], delete=ids[-2:])
    assert result['status'] == 'ok', result
    sparse_view = resumed_tools['view_landmarks']('s.tif')
    assert len(sparse_view['existing_pairs']) == 3
    before = copy.deepcopy(resumed.to_dict())
    refusal = resumed_tools['warp_landmarks']('s.tif', sparse_view['view_id'], method='spline')
    assert refusal['status'] == 'refused', refusal
    assert resumed.to_dict() == before
    added = resumed_tools['edit_landmarks']('s.tif', sparse_view['view_id'],
                                             add=pairs_for(sparse_view)[-2:])
    assert added['status'] == 'ok', added
    new_ids = [p['id'] for p in resumed.slices[0].landmark_pairs]
    assert new_ids[:3] == ids[:3]
    assert min(new_ids[-2:]) > max(ids)


def test_existing_spline_seeds_raw_source_pairs_without_mutating_on_view(env):
    tools, _, state, ctx = env
    view = tools['view_landmarks']('s.tif')
    target = RAW.copy()
    target[-1] += [11, -7]
    result = tools['warp_landmarks']('s.tif', view['view_id'], pairs_for(view, target=target),
                                     method='spline')
    assert result['status'] == 'ok', result
    state.slices[0].landmark_pairs = None
    state.slices[0].landmark_frame = None
    before = copy.deepcopy(state.to_dict())
    restarted = {fn.__name__: fn for fn in build_tools(state, ctx, ctx.spec).tools}
    seeded = restarted['view_landmarks']('s.tif')
    assert seeded['status'] == 'ok', seeded
    assert state.to_dict() == before
    assert len(seeded['existing_pairs']) == 5
    for index, pair in enumerate(seeded['existing_pairs']):
        a = image_to_section(np.array([pair['slice_xy']]), seeded['images_info'][0])[0]
        b = image_to_section(np.array([pair['atlas_xy']]), seeded['images_info'][1])[0]
        np.testing.assert_allclose(a, RAW[index], atol=1e-7)
        np.testing.assert_allclose(b, target[index], atol=1e-7)


def test_marker_numbers_appear_on_both_distinct_pair_locations(env):
    tools, _, _, _ = env
    blank = tools['view_landmarks']('s.tif')
    target = RAW.copy()
    target[-1] += [40, 35]
    result = tools['edit_landmarks']('s.tif', blank['view_id'],
                                    add=pairs_for(blank, target=target))
    assert result['status'] == 'ok', result
    marked = tools['view_landmarks']('s.tif')
    for index, coordinate in [(0, 'slice_xy'), (1, 'atlas_xy')]:
        pristine = pixels(blank[TOOL_MEDIA_PARTS_KEY][index])
        annotated = pixels(marked[TOOL_MEDIA_PARTS_KEY][index])
        x, y = np.round(marked['existing_pairs'][-1][coordinate]).astype(int)
        patch = np.s_[max(0, y - 14):y + 15, max(0, x - 14):x + 15]
        assert np.abs(pristine[patch].astype(float) - annotated[patch]).max() > 80
    assert marked['existing_pairs'][-1]['slice_xy'] != marked['existing_pairs'][-1]['atlas_xy']


def test_frame_change_requires_explicit_reset_and_prior_edit_view_is_stale(env):
    tools, _, state, _ = env
    initial = tools['view_landmarks']('s.tif')
    edit = tools['edit_landmarks']('s.tif', initial['view_id'], add=pairs_for(initial))
    assert edit['status'] == 'ok', edit
    stale = tools['edit_landmarks']('s.tif', initial['view_id'], delete=[1])
    assert stale['error'] == 'STALE_LANDMARK_VIEW'
    state.slices[0].position_mm += .025
    view = tools['view_landmarks']('s.tif')
    assert view['landmark_frame_stale']
    assert not view['existing_pairs']
    before = copy.deepcopy(state.to_dict())
    refusal = tools['edit_landmarks']('s.tif', view['view_id'], add=pairs_for(view))
    assert refusal['error'] == 'STALE_LANDMARK_FRAME'
    assert state.to_dict() == before
    fresh = tools['edit_landmarks']('s.tif', view['view_id'], reset=True,
                                    add=pairs_for(view))
    assert fresh['status'] == 'ok', fresh
    assert len(state.slices[0].landmark_pairs) == 5
    assert not tools['view_landmarks']('s.tif')['landmark_frame_stale']


def test_deleting_all_pairs_does_not_resurrect_applied_spline_landmarks(env):
    tools, _, state, _ = env
    view = tools['view_landmarks']('s.tif')
    target = RAW.copy()
    target[-1] += [6, 8]
    result = tools['warp_landmarks']('s.tif', view['view_id'], pairs_for(view, target=target),
                                     method='spline')
    assert result['status'] == 'ok', result
    applied = copy.deepcopy(state.slices[0].transform)
    marked = tools['view_landmarks']('s.tif')
    cleared = tools['edit_landmarks']('s.tif', marked['view_id'],
                                       delete=[p['id'] for p in marked['existing_pairs']])
    assert cleared['status'] == 'ok', cleared
    assert state.slices[0].landmark_pairs == []
    assert state.slices[0].transform == applied
    empty = tools['view_landmarks']('s.tif')
    assert empty['existing_pairs'] == []


@pytest.mark.parametrize('count', [3, 5])
def test_default_landmark_method_fits_full_affine_with_shear(env, count):
    from langslice.affine import denormalized_affine

    tools, _, state, _ = env
    matrix = np.array([[1.03, .11, -12], [-.025, .94, 17]])
    source = RAW[:count]
    target = source @ matrix[:, :2].T + matrix[:, 2]
    view = tools['view_landmarks']('s.tif')
    result = tools['warp_landmarks']('s.tif', view['view_id'], pairs_for(view, source, target))
    assert result['status'] == 'ok', result
    transform = state.slices[0].transform
    assert transform['kind'] == 'interactive'
    assert transform.get('spline') is None
    np.testing.assert_allclose(denormalized_affine(transform['params'], (512, 512)),
                               matrix, atol=1e-9)
    assert abs(transform['physical']['shear']) > .01
    assert max(result['landmark_residuals_mm']) < 1e-8
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 4
    assert [pair['id'] for pair in result['existing_pairs']] == [
        pair['id'] for pair in state.slices[0].landmark_pairs]


def test_same_persistent_points_switch_affine_spline_affine_without_composition(env):
    from langslice.affine import denormalized_affine

    tools, _, state, _ = env
    target = RAW @ np.array([[1.01, -.03], [.06, .98]]) + [-9, 11]
    target[-1] += [12, -8]
    view = tools['view_landmarks']('s.tif')
    edit = tools['edit_landmarks']('s.tif', view['view_id'],
                                    add=pairs_for(view, target=target))
    assert edit['status'] == 'ok', edit
    draft = copy.deepcopy(state.slices[0].landmark_pairs)
    # Ordinary least squares is an independent expected full affine, including shear.
    expected = np.linalg.lstsq(np.column_stack((RAW, np.ones(len(RAW)))), target, rcond=None)[0].T
    affine = tools['warp_landmarks']('s.tif', edit['view_id'])
    assert affine['status'] == 'ok', affine
    first_params = copy.deepcopy(state.slices[0].transform['params'])
    np.testing.assert_allclose(denormalized_affine(first_params, (512, 512)), expected, atol=1e-9)
    view = tools['view_landmarks']('s.tif')
    spline = tools['warp_landmarks']('s.tif', view['view_id'], method='spline')
    assert spline['status'] == 'ok', spline
    assert state.slices[0].landmark_pairs == draft
    stored = state.slices[0].transform['spline']
    np.testing.assert_allclose(stored['source'], RAW / 512, atol=1e-10)
    np.testing.assert_allclose(stored['target'], target / 512, atol=1e-10)
    np.testing.assert_allclose(fit_spline(stored).forward(RAW * .01), target * .01, atol=2e-6)
    view = tools['view_landmarks']('s.tif')
    restored = tools['warp_landmarks']('s.tif', view['view_id'], method='affine')
    assert restored['status'] == 'ok', restored
    assert state.slices[0].transform.get('spline') is None
    np.testing.assert_allclose(state.slices[0].transform['params'], first_params, atol=1e-12)
    assert state.slices[0].landmark_pairs == draft


@pytest.mark.parametrize('failure', ['too_few', 'collinear', 'duplicate', 'reflection', 'unknown'])
def test_bad_affine_pairs_are_refused_before_state_or_draft_write(env, failure):
    tools, box, state, _ = env
    source = RAW.copy()
    target = RAW.copy()
    method = 'affine'
    if failure == 'too_few':
        source, target = source[:2], target[:2]
    elif failure == 'collinear':
        source = np.array([[100, 100], [200, 200], [300, 300]])
        target = source.copy()
    elif failure == 'duplicate':
        source[1] = source[0]
    elif failure == 'reflection':
        target[:, 0] = 512 - target[:, 0]
    else:
        method = 'elastic_magic'
    view = tools['view_landmarks']('s.tif')
    before = copy.deepcopy(state.to_dict())
    result = tools['warp_landmarks']('s.tif', view['view_id'],
                                     pairs_for(view, source, target), method=method)
    assert result['status'] == 'refused', result
    assert state.to_dict() == before
    assert not box.undo_stack


def test_direct_apply_returns_numbered_points_on_raw_and_atlas_feedback(env):
    tools, _, _, _ = env
    view = tools['view_landmarks']('s.tif')
    target = RAW.copy()
    target[-1] += [35, 25]
    result = tools['warp_landmarks']('s.tif', view['view_id'], pairs_for(view, target=target))
    assert result['status'] == 'ok', result
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 4
    refreshed = tools['view_landmarks']('s.tif')
    for feedback_index, reference_index in [(2, 0), (3, 1)]:
        returned = pixels(result[TOOL_MEDIA_PARTS_KEY][feedback_index])
        reference = pixels(refreshed[TOOL_MEDIA_PARTS_KEY][reference_index])
        np.testing.assert_array_equal(returned, reference)
        blank = pixels(view[TOOL_MEDIA_PARTS_KEY][reference_index])
        point = result['existing_pairs'][-1]
        coordinate = point['slice_xy' if reference_index == 0 else 'atlas_xy']
        x, y = np.round(coordinate).astype(int)
        patch = np.s_[max(0, y - 14):y + 15, max(0, x - 14):x + 15]
        assert np.abs(returned[patch].astype(float) - blank[patch]).max() > 80
