# Visual context lifecycle — shelved design record

Status: acceptance-based implementation removed, 2026-09-13. The sections
under “Historical proposal” below preserve the design considered on 2026-09-12;
they are not instructions for the current runtime.

## Current direction

Keep the established `WorkingSetImages` policy unchanged: protect the seed,
perform the first transform-media stage cut, and trim at a 256-image high /
128-image low working set. The only supported `image_retention` value is
`legacy`; requesting the removed `completion` policy fails explicitly.

There is no `accept_views` tool or added final-view inspection requirement.
The model chooses its scientific strategy without managing image acceptance.
Predictive cost triggers are shelved, not implemented. Preserve low-resolution
global context, higher-resolution inspection, separate positioning references,
and delivery-aware suppression of already-seen placement pictures.

Optimize image delivery before adding pruning machinery. Accounting must
distinguish newly appended image/text input, cached replay, and uncached replay
after history edits. Aggregate usage alone cannot establish that split; missing
attribution is unknown, not zero, and attachment dimensions are not token bills.
Only evidence of substantial retained-image carry cost should motivate a new
retention policy. Test cache mechanics with fixed histories first, then test
agent behavior independently with quality and strategy changes controlled.
Keep final registration quality separate from cost and inspect overlays as well
as numerical scores when evaluating registration.

## Historical proposal (not active)

The following proposed completion-based retirement was an opt-in replacement
for the established broad cuts, not a description of the current policy.

## Objective

Keep the whole stack visible from the beginning, retain the actual textual
trajectory, and keep only accepted visual results after exploration is done.
Detailed inspection of one section can reveal position errors, cutting-angle
changes, hemisphere flips, or evidence about neighboring sections. That work
must remain in the main agent conversation, including when independent
sections are adjusted in a batch. There are no alignment forks, subagents,
separate sessions, or summary-and-merge steps in this design.

For a completed positioning-and-transform run, the intended retained context
is:

```text
Initial low-resolution section images and atlas reference images
Positioning tools, reasoning, and text

Section A: transform tools, reasoning, and text
Section A: accepted final overlay

Section B: transform tools, reasoning, and text
Section B: accepted final overlay
...
```

The illustration groups sections for readability. Actual calls can be
interleaved; preserve their chronological order. Only media attachments are
retired. Preserve all existing textual content, tool calls and results, and
reasoning items. Do not replace them with a newly written textual account.

The initial dump and accepted final overlays are the persistent images in
this completed trajectory. Temporary images may be numerous while work is
active. The final retained context is not the same as the full record of
everything the agent was shown during execution.

## Image lifecycle

| Image | While work is active | After acceptance |
| --- | --- | --- |
| Initial section and atlas images | Retain unchanged | Retain unchanged |
| Intermediate transform overlays | Append normally; available for comparison | Retire together at the completion boundary |
| Accepted final transform overlay | Must have reached the agent and been inspected | Retain in its chronological position |
| Position candidates and comparison panels | Retain while choosing positions | Retain the accepted comparison until its evidence is superseded |
| Shared atlas fetches and stack review images | Retain while relevant work is unresolved | Retire at an appropriate completed batch or stage boundary |

For position-only runs, the accepted comparison is the final visual result
per section. For runs that proceed through transform, final comparisons and
other positioning views are temporary evidence; retire them once the
relevant decisions are settled and final transform overlays supersede them.
This extension needs validation before replacing the current stage cut.

### Transform completion

1. Append adjustments, their text, reasoning, and feedback images normally.
   The agent can inspect earlier overlays while refining the current one.
2. The agent inspects and accepts the resulting alignment. A successful tool
   call, a write, or a numeric overlap score alone is not acceptance.
3. Remove superseded intermediate attachments in one pass, retaining the
   accepted overlay and all existing non-image content.
4. Continue in the same conversation from this cleaned history.

For independent sections adjusted together, prefer retiring their
intermediate images once the batch is accepted. Cleaning one section while
the others are still interleaved can repeatedly disturb the same suffix.
Do not impose a one-image-per-turn or one-turn lifetime rule.

The acceptance signal is `accept_views(slice_ids, stage)`, where stage names
the kind of evidence (`position` or `transform`), not a session transition.
The model chooses its grouping and may interleave or revisit either kind of
work. Acceptance changes no scientific state. `submit` accepts all final
views, so separate acceptance calls are useful for retiring intermediate
evidence during a run, not mandatory for every section.

Only a full current view included in an earlier model request is eligible.
For transforms this is an overlay (including the current side of an A/B
view), not a zoom or a section-only image. Completion-mode `validate` and
`submit` refuse final geometry lacking this evidence. That inspection
requirement is an additional behavior of this opt-in policy; it must be
declared when comparing against the unchanged legacy harness. Delivery is
observable; human-like attention or understanding cannot be verified by the
harness. The model's acceptance is the attestation of inspection.

Attachment identities include exact response slots, section ownership,
geometry, eligibility, and delivery state. A separate-image comparison
retains its required section companion as well as its accepted atlas image.
Retirement never deletes a newly rendered sibling result before its first
delivery, even when acceptance occurs in the same model round. If submission
has unread sibling results, they remain as a conservative exception to the
seed-plus-finals target. Completion mode refuses later tool calls once
submission succeeds, so a queued sibling write cannot invalidate the accepted
final geometry. A new session is required to modify a submitted job.

### Positioning and shared evidence

A section may have several candidate comparisons before a position is
accepted. Keep the candidate evidence during that search, then retain the
comparison supporting the accepted position and retire superseded panels
together. A shared atlas fetch can inform several sections; finishing one
section does not establish that this reference is no longer needed.
The implementation conservatively keeps generic shared evidence until every
section is accepted for the final enabled task: transform when enabled,
otherwise position. Completing positioning alone cannot remove references
that may still inform interleaved transform work. Do not guess ownership from
the latest tool call.

An already-seen placement can suppress a duplicate picture on write only
when the relevant section, position, orientation, and cutting angles match.
Images not delivered to a prior model request do not count as seen. A
comparison and write requested in the same model round are not an
inspect-then-accept sequence. Successful rendering is distinct from delivery,
and historical delivery is distinct from continued presence in context.

### Reopening a decision

The agent can revisit an accepted section. New parameters, orientation, or
cutting angles may make its retained overlay stale. Append fresh evidence
while reopening the work; do not silently present the old overlay as current.
When the revised result is accepted, retire the superseded accepted image
along with the new intermediate images. This restores the target of one final
overlay per section but can require rebuilding a much longer suffix. Include
these revisits in cost and accuracy evaluation.

## Cache behavior and economics

Prefix caching reuses an unchanged beginning of a request. Removing an image
does not flush all cached state: it prevents the edited request from matching
the old continuation at and after the first edit. The initial image dump and
previously completed blocks before that edit remain eligible for reuse.

```text
Before: [seed][completed blocks][text][temporary image][text][final image]
After:  [seed][completed blocks][text]                 [text][final image]
        <------ reusable prefix --->                 <- rebuilt suffix ->
```

Actual reuse ends at an available matching cache boundary, not necessarily
the exact preceding token. Retention, routing, model settings, and provider
behavior also affect whether the expected prefix is available.

The public GPT-5.6 documentation describes implicit lookup over up to 20
earlier eligible message endings. An unchanged seed can therefore fall
outside the lookup range during a long active block. Explicit breakpoints
can preserve older lookup boundaries on the public API, but support on the
subscription backend has not been established. This implementation changes
image retention only; it does not send new cache-control fields. Test long
blocks and early revisits rather than assuming seed-cache reuse from byte
identity alone.

The next request processes the surviving edited suffix and can establish a
new reusable prefix for subsequent requests. This is one intended cache
disruption per completed block, not a guarantee of exactly one paid rebuild
under every provider condition. Multiple cached continuations may coexist;
they cannot be stitched into a request from independently cached fragments.
No explicit fork or merge operation is required.

The tradeoff is the future discounted carry cost of the removed images versus
the cost of rebuilding the surviving suffix, including any cache-write cost.
The unchanged initial image dump is outside that comparison when its prefix
still hits. Early edits, interleaved work, and later revisits increase the
suffix that needs rebuilding. Do not assume pruning is economical solely
because it reduces the raw input count.

Public API prices are not a substitute for measured subscription usage on the
OAuth transport. Record actual cached and uncached input and available quota
measurements. An image absent from future requests is absent from future
model context; this policy makes no claim about provider-side data retention.

## Reasoning and audit requirements

Preserving every reasoning item in the client transcript does not establish
that the provider still accepts and renders it after an earlier image is
removed. The current OAuth path replays encrypted reasoning. Validate its
continuity explicitly before enabling this policy. Do not silently drop
reasoning, substitute summaries, or claim that retaining its bytes proves it
remains usable.

Keep the execution trace faithful to what was delivered at each step. Record
retirement events and image identities separately from original textual
results, so an attachment described earlier is distinguishable from one
currently available. Preserve the original trace content; do not rewrite the
historical text to make the cleaned request look like the original execution.
An emergency context-limit policy must remain explicit and must not silently
override these retention guarantees during ordinary runs.

Completion mode does not also run the legacy stage/count filter. Unaccepted
work can grow without an image-count cut; the existing optional single-request
input safeguard is independent. The registry is session-local, not a new
checkpoint format: a resumed job is seeded afresh, as before. Original ADK
history remains intact; only outgoing request copies lose retired attachments.
Preservation checks compare before/after requests through the same serializer;
they do not establish that the existing transport reproduces every original
provider output field or that encrypted reasoning remains semantically intact.

## Validation before rollout

Use repeated runs across representative stacks, including damaged sections,
orientation corrections, cutting-angle changes, and revisited decisions.
Retirement does not require a separate model sampling step; choosing its
policy does require empirical evidence across agent variability.

Vary one factor at a time:

- **Retention:** compare the current policy with completion-based retirement,
  holding rendering, model settings, and task inputs constant. An unchanged
  legacy control measures the whole completion-workflow bundle (acceptance
  tool, inspection requirement, and retirement). To isolate image retirement,
  both arms must instead expose identical acceptance tools and requirements.
  Do not label that modified control as the unchanged current harness.
- **Presentation:** compare the persistent seed section plus a separately
  supplied atlas with a side-by-side comparison, matching the constituent
  image resolutions and retaining the same context policy. Early user tests
  favored separate images; this is motivation to test, not a settled result.
- **Resolution:** evaluate lower-resolution positioning separately from more
  detailed interactive-transform feedback. The caps and any crop policy
  remain to be determined; do not reintroduce arbitrary upscaling.

Measure position error, alignment quality, model rounds, corrective work,
reopened sections, failures, raw input, cached input, uncached input, and
actual spend or quota usage. Inspect the resulting placements and overlays
alongside the metrics; report disagreements rather than hiding them behind a
score. Compare distributions over repeated runs, not a single favorable
trajectory. Establish the baseline after the image-upscaling fix.

Transport checks must separately establish that the seed still hits after a
retirement event, later requests reuse the cleaned prefix, all text survives,
reasoning remains usable, and batched results preserve their final images.

## Research basis

- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching):
  matching prefixes and cache boundaries; editing a continuation does not
  necessarily discard the earlier cache entry.
- [Anthropic computer-use best practices](https://claude.com/blog/best-practices-for-computer-and-browser-use-with-claude):
  batch screenshot pruning amortizes cache disruption; a small post-pruning
  retained count is not a strict image limit on every turn.
- [Anthropic context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing):
  even server-side tool-result clearing has a cache-rebuild tradeoff.
- [Browser Use message management](https://github.com/browser-use/browser-use/blob/main/browser_use/agent/message_manager/service.py)
  and [prompt construction](https://github.com/browser-use/browser-use/blob/main/browser_use/agent/prompts.py):
  a contrasting design reconstructs textual history before a current
  screenshot. LangSlice's chosen design instead preserves the chronological
  conversation and retires images at completion boundaries.
- [OpenAI reasoning](https://developers.openai.com/api/docs/guides/reasoning):
  documented reasoning replay carries history forward; compatibility with
  selective image removal requires a transport-specific check.

Sources inspected during the 2026-09-10 design discussion. These examples
motivate the experiments; none establishes registration accuracy or savings
for LangSlice.
