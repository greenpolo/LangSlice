"""The agent CLI's fix sentence per refusal code."""

from __future__ import annotations

#: What to do about a code (the envelope's ``error.fix``), written with what
#: it needs (:mod:`langslice.job.settings.needs`); ``{job}`` is the job folder,
#: ``{verb}`` the verb.
FIXES: dict[str, str] = {
    "UNKNOWN_VERB": "List the verbs with `langslice-job ops`.",
    "VERB_OFF": "This job's tasks do not include this verb; `langslice-job {job} status` "
                "lists the job's verbs under `verbs`.",
    "UNKNOWN_ARGUMENTS": "Check the argument names with `langslice-job schema {verb}`.",
    "MISSING_ARGUMENTS": "Give every required argument; see `langslice-job schema {verb}`.",
    "BAD_JSON": "Pass --args a JSON object, or @path/to/file.json.",
    "UNKNOWN_SLICE_IDS": "Name sections by filename, as `langslice-job {job} status` "
                         "lists them.",
    "UNKNOWN_SECTION": "Name sections by filename, as `langslice-job {job} status` "
                       "lists them.",
    "NO_JOB": "Create the job first: `langslice-job <image folder> init`; a job made "
              "with --job-dir is named by its own folder.",
    "NO_IMAGES": "Point init at a folder of section images (TIFF, PNG or JPEG).",
    "NOTHING_TO_UNDO": "The job's history holds no earlier step.",
    "BAD_PATH": "Give a path inside the job folder, relative to it.",
    "NOT_FOUND": "The message names the nearest folder above the path that exists and what "
                 "it holds; give a path from there[[?list_files  (list_files lists it)]].",
    "NOTHING_TO_REDO": "Redo follows an undo only.",
    "UNKNOWN_REGISTRATION": "List the section's registrations[[?restore_registration : "
                            "`langslice-job {job} restore_registration --section NAME`]].",
    "REGISTRATION_ELSEWHERE": "Put the section back at the position and cutting angles the "
                              "message names, then call it again.",
    "NO_POSITION": "Write the section's position first[[?position_sections  "
                   "(position_sections)]].",
    "MISSING_POSITIONS": "Write every section's own position first[[?position_sections  "
                         "(position_sections)]].",
    "MISSING_TRANSFORMS": "Give every section a transform[[?elastix_affine|"
                          "interactive_transform  ([[%elastix_affine=elastix_affine||"
                          "interactive_transform=interactive_transform]])]].",
    "MISSING_DEFORMATIONS": "Give every section a deformation[[?fireants_syn|trace_borders  "
                            "([[%fireants_syn=fireants_syn||trace_borders=trace_borders]])]].",
    "FROZEN": "The user froze these properties of these sections (the reply names them): "
              "leave them out of the call.",
    "STALE_INPUT": "The section changed while this ran; run the verb again for it.",
    "SECTION_BUSY": "Background work is running on this section (the reply names it); "
    "write to it once the work lands.",
    "JOB_BUSY": "Another writer held the job folder's lock; retry.",
    "NOTHING_FITTED": "Each section's row under result.results names its problem.",
    "NOTHING_WRITTEN": "Each section's row under result.results names its problem.",
    "UNKNOWN_RUN": "List the background runs with `langslice-job {job} runs`.",
    "RUN_LOST": "The background process ended without an answer; see its log under "
                "logs/runs/ and run the verb again. If every background run is lost, "
                "your shell ends the processes a command leaves running when it returns "
                "(sandboxes do this): run the verb as `LANGSLICE_DETACH=0 langslice-job "
                "...`, without --background, in your harness's own background mode if it "
                "has one.",
    "BAD_REGISTRATION": "Give --registration a QuickNII/VisuAlign JSON or XML, a DeepSlice "
                        "CSV/JSON/XML or a LangSlice registration.json whose entries name "
                        "this folder's section images (by file name, stem or _sNNN number) "
                        "on the job's atlas.",
    "INPUTS_CHANGED": "Rerun `langslice-job {job} init` with --fresh to start a new job "
                      "from these inputs (the old one's work is replaced), or with the "
                      "inputs the job was made from (job.json, spec.inputs) to continue it.",
    "IMAGE_MODEL_OFF": "Connect the job's image model to LangSlice (`langslice login` for "
                       "openai-oauth, or its API key) and call it again[[?fireants_syn ; "
                       "without it, fit each section's deformation to its stain with "
                       "fireants_syn]].",
    "NOTHING_TRACED": "Each section's row under result.results names its problem"
                      "[[?trace_borders ; a section needs a position and a transform"
                      "[[?elastix_affine|interactive_transform  ([[%elastix_affine="
                      "elastix_affine||interactive_transform=interactive_transform]])]] "
                      "before trace_borders]].",
    "INTERVAL_BREAKS_UNSUPPORTED": "Report a break only where the written spacing exceeds "
                                   "1.5x the stack's median spacing; check the positions "
                                   "on both sides first.",
    "STRICT_INTERVAL": "Make every spacing within 10% of the nominal interval and report "
                       "no break (the user asked for a strict interval).",
    "OPTION_OFF": "That option is switched off for this job; use one of the others the "
                  "message lists.",
    "FLIP_DISABLED": "Flipping is switched off for this job; leave flip out.",
    "INVALID_LINEAR_PLACEMENT": "Give the section a position and a transform"
                                "[[?elastix_affine|interactive_transform  ([[%elastix_affine="
                                "elastix_affine||interactive_transform=interactive_transform"
                                "]])]] first.",
    "ENGINE_MISSING": "torch or fireants does not import on this host; reinstall "
                    "LangSlice (see the message).",
    "FIT_FAILED": "The engine failed on this section: the message "
                  "says why, and logs/engine.log holds the whole error. When the GPU ran "
                  "out of memory, run the call again.",
    "RENDER_FAILED": "See the message; check the section file opens, then retry.",
    "UNKNOWN_MASK": "Make the mask first[[?make_mask  with make_mask]]; `status` lists each "
                    "section's masks.",
    "STALE_MASK": "The section was turned or flipped since this mask was made; make the mask again"
                  "[[?make_mask  with make_mask]].",
    "UNKNOWN_IMAGE": "`status` lists the section images[[?make_image ; make one with "
                     "make_image]].",
    "MASK_UNSUPPORTED": "This engine sorts tissue itself and takes no mask; the message names "
                        "the engines that do.",
    "NO_DEFORMATION": "The section holds no deformation; fit one first, or make the mask "
                      "with source \"em\".",
    "NO_SORTING": "The section's deformation saved no sorting; make the mask with source "
                  "\"em\".",
    "HAS_DEFORMATION": "Undo the deformation first, or fit a warp on it.",
    "FIT_ATLAS_UNAVAILABLE": "Use atlas_image \"template\" (the reference template).",
    "NO_PICTURE": "Look first[[?zoom : zoom takes a box on a picture already drawn]].",
    "UNKNOWN_PICTURE": "Use a picture number from a reply's `pictures`, or views.jsonl.",
    "PICTURE_NOT_SAVED": "This job keeps no pictures between calls; draw it again with look"
                         "[[?zoom  and zoom]] in the same process.",
    "EMPTY_BOX": "Give a box with area inside the picture: [x0, y0, x1, y1] in its pixels.",
    "TOO_MANY_LANDMARK_VIEWS": "Nothing was written. Send the same pairs with `--view false`, "
                               "or in calls of four pairs or fewer.",
    "BAD_LANDMARK_POINT": "Give x,y inside the image content the message states[[?zoom  (on a "
                          "zoom, numbers its grid prints)]].",
    "ATLAS_ENDPOINT_OUTSIDE": "Nothing was written. Send the pairs again with each atlas "
                              "endpoint the message names at the point inside the atlas it "
                              "gives, or on the drawn border where you meant it.",
    "BAD_LANDMARK_PICTURE":"Draw this section with look (sections=[its filename]) and give "
                            "that picture's number; the message says which sections the "
                            "picture you named covers.",
    "STALE_LANDMARK_PLANE": "Draw the section's current atlas plane with look, then re-place "
                            "the pairs the message names (their atlas endpoints), or delete "
                            "them.",
    "UNKNOWN_LANDMARK": "Use a pair id the message lists[[?set_landmarks : set_landmarks "
                        "with only the section lists the pairs]].",
    "INSUFFICIENT_LANDMARKS": "Add pairs[[?set_landmarks  with set_landmarks]] until the "
                              "section has at least three, then apply again.",
    "LANDMARK_TRIANGULATION_FAILED": "Move or delete the pairs the message names"
                                     "[[?set_landmarks  with set_landmarks]], then apply "
                                     "again.",
    "NOTHING_EXPORTED": "Each skipped section's reason is under result.skipped; place it "
                        "first.",
}
