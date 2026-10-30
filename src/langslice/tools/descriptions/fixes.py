"""The agent CLI's fix sentence per refusal code."""

from __future__ import annotations

from langslice.core.refusal import RefusalCode

#: What to do about a code (the envelope's ``error.fix``), written with what
#: it needs (:mod:`langslice.job.settings.needs`); ``{job}`` is the job folder,
#: ``{verb}`` the verb.
FIXES: dict[str, str] = {
    RefusalCode.UNKNOWN_VERB: "List the verbs with `langslice-job ops`.",
    RefusalCode.VERB_OFF: (
        "This job's tasks do not include this verb; `langslice-job {job} status` "
        "lists the job's verbs under `verbs`."
    ),
    RefusalCode.UNKNOWN_ARGUMENTS: "Check the argument names with `langslice-job schema {verb}`.",
    RefusalCode.MISSING_ARGUMENTS: (
        "Give every required argument; see `langslice-job schema {verb}`."
    ),
    RefusalCode.BAD_JSON: "Pass --args a JSON object, or @path/to/file.json.",
    RefusalCode.UNKNOWN_SLICE_IDS: (
        "Name sections by filename, as `langslice-job {job} status` "
        "lists them."
    ),
    RefusalCode.SECTION_GONE: (
        "The picture shows a section the stack no longer holds; "
        "`langslice-job {job} status` lists the sections it holds now."
    ),
    RefusalCode.NO_JOB: (
        "Create the job first: `langslice-job <image folder> init`; a job made "
        "with --job-dir is named by its own folder."
    ),
    RefusalCode.NO_IMAGES: "Point init at a folder of section images (TIFF, PNG or JPEG).",
    RefusalCode.NOTHING_TO_UNDO: "The job's history holds no earlier step.",
    RefusalCode.BAD_PATH: "Give a path inside the job folder, relative to it.",
    RefusalCode.NO_SUCH_PATH: (
        "The message names the nearest folder above the path that exists and what "
        "it holds; give a path from there[[?list_files  (list_files lists it)]]."
    ),
    RefusalCode.NOTHING_TO_REDO: "Redo follows an undo only.",
    RefusalCode.UNKNOWN_REGISTRATION: (
        "List the section's registrations[[?restore_registration : "
        "`langslice-job {job} restore_registration --section NAME`]]."
    ),
    RefusalCode.REGISTRATION_ELSEWHERE: (
        "Put the section back at the position and cutting angles the "
        "message names, then call it again."
    ),
    RefusalCode.NO_POSITION: (
        "Write the section's position first[[?position_sections  "
        "(position_sections)]]."
    ),
    RefusalCode.MISSING_POSITIONS: (
        "Write every section's own position first[[?position_sections  "
        "(position_sections)]]."
    ),
    RefusalCode.MISSING_TRANSFORMS: (
        "Give every section a transform[[?elastix_affine|"
        "interactive_transform  ([[%elastix_affine=elastix_affine||"
        "interactive_transform=interactive_transform]])]]."
    ),
    RefusalCode.MISSING_DEFORMATIONS: (
        "Give every section a deformation[[?fireants_syn|trace_borders  "
        "([[%fireants_syn=fireants_syn||trace_borders=trace_borders]])]]."
    ),
    RefusalCode.FROZEN: (
        "The user froze these properties of these sections (the reply names them): "
        "leave them out of the call."
    ),
    RefusalCode.STALE_INPUT: "The section changed while this ran; run the verb again for it.",
    RefusalCode.SECTION_BUSY: (
        "Background work is running on this section (the reply names it); "
        "write to it once the work lands."
    ),
    RefusalCode.JOB_BUSY: "Another writer held the job folder's lock; retry.",
    RefusalCode.NOTHING_FITTED: "Each section's row under result.results names its problem.",
    RefusalCode.NOTHING_WRITTEN: "Each section's row under result.results names its problem.",
    RefusalCode.UNKNOWN_RUN: "List the background runs with `langslice-job {job} runs`.",
    RefusalCode.RUN_LOST: (
        "The background process ended without an answer; see its log under "
        "logs/runs/ and run the verb again. If every background run is lost, "
        "your shell ends the processes a command leaves running when it returns "
        "(sandboxes do this): run the verb as `LANGSLICE_DETACH=0 langslice-job "
        "...`, without --background, in your harness's own background mode if it "
        "has one."
    ),
    RefusalCode.BAD_REGISTRATION: (
        "Give --registration a QuickNII/VisuAlign JSON or XML, a DeepSlice "
        "CSV/JSON/XML or a LangSlice registration.json whose entries name "
        "this folder's section images (by file name, stem or _sNNN number) "
        "on the job's atlas."
    ),
    RefusalCode.INPUTS_CHANGED: (
        "Rerun `langslice-job {job} init` with --fresh to start a new job "
        "from these inputs (the old one's work is replaced), or with the "
        "inputs the job was made from (job.json, spec.inputs) to continue it."
    ),
    RefusalCode.IMAGE_MODEL_OFF: (
        "Connect the job's image model to LangSlice (`langslice login` for "
        "openai-oauth, or its API key) and call it again[[?fireants_syn ; "
        "without it, fit each section's deformation to its stain with "
        "fireants_syn]]."
    ),
    RefusalCode.INTERVAL_BREAKS_UNSUPPORTED: (
        "Report a break only where the written spacing exceeds "
        "1.5x the stack's median spacing; check the positions "
        "on both sides first."
    ),
    RefusalCode.STRICT_INTERVAL: (
        "Make every spacing within 10% of the nominal interval and report "
        "no break (the user asked for a strict interval)."
    ),
    RefusalCode.OPTION_OFF: (
        "That option is switched off for this job: leave it out, or use one "
        "of the others the message lists."
    ),
    RefusalCode.INVALID_LINEAR_PLACEMENT: (
        "Give the section a position and a transform"
        "[[?elastix_affine|interactive_transform  ([[%elastix_affine="
        "elastix_affine||interactive_transform=interactive_transform"
        "]])]] first."
    ),
    RefusalCode.ENGINE_MISSING: (
        "The engine is not installed on this host (the message names what is "
        "missing): use another method, or ask the user to install it."
    ),
    RefusalCode.FIT_FAILED: (
        "The engine failed on this section: the message "
        "says why, and logs/engine.log holds the whole error. When the GPU ran "
        "out of memory, run the call again."
    ),
    RefusalCode.RENDER_FAILED: "See the message; check the section file opens, then retry.",
    RefusalCode.UNKNOWN_MASK: (
        "Make the mask first[[?make_mask  with make_mask]]; `status` lists each "
        "section's masks."
    ),
    RefusalCode.STALE_MASK: (
        "The section was turned or flipped since this mask was made; make the mask again"
        "[[?make_mask  with make_mask]]."
    ),
    RefusalCode.UNKNOWN_IMAGE: (
        "`status` lists the section images[[?make_image ; make one with "
        "make_image]]."
    ),
    RefusalCode.MASK_UNSUPPORTED: (
        "This engine sorts tissue itself and takes no mask; the message names "
        "the engines that do."
    ),
    RefusalCode.NO_DEFORMATION: (
        "The section holds no deformation; fit one first, or make the mask "
        "with source \"em\"."
    ),
    RefusalCode.NO_SORTING: (
        "The section's deformation saved no sorting; make the mask with source "
        "\"em\"."
    ),
    RefusalCode.HAS_DEFORMATION: "Undo the deformation first, or fit a warp on it.",
    RefusalCode.NO_PICTURE: "Look first[[?zoom : zoom takes a box on a picture already drawn]].",
    RefusalCode.UNKNOWN_PICTURE: "Use a picture number from a reply's `pictures`, or views.jsonl.",
    RefusalCode.PICTURE_NOT_SAVED: (
        "This job keeps no pictures between calls; draw it again with look"
        "[[?zoom  and zoom]] in the same process."
    ),
    RefusalCode.EMPTY_BOX: (
        "Give a box with area inside the picture: [x0, y0, x1, y1] in its pixels."
    ),
    RefusalCode.TOO_MANY_LANDMARK_VIEWS: (
        "Nothing was written. Send the same pairs with `--view false`, "
        "or in calls of four pairs or fewer."
    ),
    RefusalCode.BAD_LANDMARK_POINT: (
        "Give x,y inside the image content the message states[[?zoom  (on a "
        "zoom, numbers its grid prints)]]."
    ),
    RefusalCode.ATLAS_ENDPOINT_OUTSIDE: (
        "Nothing was written. Send the pairs again with each atlas "
        "endpoint the message names at the point inside the atlas it "
        "gives, or on the drawn border where you meant it."
    ),
    RefusalCode.BAD_LANDMARK_PICTURE: (
        "Draw this section with look (sections=[its filename]) "
        "and give that picture's number; the message says which "
        "sections the picture you named covers."
    ),
    RefusalCode.STALE_LANDMARK_PLANE: (
        "Draw the section's current atlas plane with look, then re-place "
        "the pairs the message names (their atlas endpoints), or delete "
        "them."
    ),
    RefusalCode.UNKNOWN_LANDMARK: (
        "Use a pair id the message lists[[?set_landmarks : set_landmarks "
        "with only the section lists the pairs]]."
    ),
    RefusalCode.INSUFFICIENT_LANDMARKS: (
        "Add pairs[[?set_landmarks  with set_landmarks]] until the "
        "section has at least three, then apply again."
    ),
    RefusalCode.LANDMARK_TRIANGULATION_FAILED: (
        "Move or delete the pairs the message names"
        "[[?set_landmarks  with set_landmarks]], then apply "
        "again."
    ),
    RefusalCode.NOTHING_EXPORTED: (
        "Each skipped section's reason is under result.skipped; place it "
        "first."
    ),
}
