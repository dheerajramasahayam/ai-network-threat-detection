# IEEE Submission Package

This directory contains a ready-to-upload package for IEEE journal submission based on the current manuscript and repository artifacts.

## Recommended Upload Mapping

- `Main Manuscript`:
  - `submission/drift_adaptive_intrusion_detection_main_manuscript.zip`
- `Conflict of Interest`:
  - `submission/conflict_of_interest_statement.pdf`
  - the manuscript itself also contains the required no-conflict disclosure
- `Supplementary Material for Review`:
  - `submission/drift_adaptive_intrusion_detection_supplementary_review.zip`
- `Cover letter / Comments`:
  - `submission/cover_letter.pdf`

## Included Source Files

- `build_submission_assets.sh`: rebuilds PDFs and archive bundles
- `conflict_of_interest_statement.tex`: no-conflict statement source
- `cover_letter.tex`: cover letter source
- `main_manuscript_bundle/`: packaged LaTeX manuscript files
- `supplementary_review/`: reviewer-facing supplementary package

## Notes

- The cover letter currently targets `IEEE Transactions on Network and Service Management`, which is the recommended fit for this paper.
- If you submit to a different IEEE venue, update only the journal name and first paragraph in `cover_letter.tex`, then rerun `bash submission/build_submission_assets.sh`.
- The manuscript bundle is source-first because IEEE submission portals commonly request a single LaTeX archive for the main manuscript.
