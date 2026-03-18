#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUBMISSION_DIR="$ROOT_DIR/submission"
MAIN_BUNDLE_DIR="$SUBMISSION_DIR/main_manuscript_bundle"
SUPPLEMENTARY_DIR="$SUBMISSION_DIR/supplementary_review"

build_tex_pdf() {
  local tex_dir="$1"
  local tex_file="$2"

  if command -v tectonic >/dev/null 2>&1; then
    (cd "$tex_dir" && tectonic "$tex_file" --outdir .)
    return
  fi

  if ! command -v pdflatex >/dev/null 2>&1; then
    echo "Either tectonic or pdflatex is required." >&2
    exit 1
  fi

  local stem="${tex_file%.tex}"
  (
    cd "$tex_dir"
    pdflatex "$tex_file"
    pdflatex "$tex_file"
    if [ -f "${stem}.aux" ] && grep -q "\\citation" "${stem}.aux" && command -v bibtex >/dev/null 2>&1; then
      bibtex "$stem"
      pdflatex "$tex_file"
      pdflatex "$tex_file"
    fi
  )
}

mkdir -p "$MAIN_BUNDLE_DIR" "$SUPPLEMENTARY_DIR"

bash "$ROOT_DIR/paper/build.sh"
build_tex_pdf "$SUBMISSION_DIR" "conflict_of_interest_statement.tex"
build_tex_pdf "$SUBMISSION_DIR" "cover_letter.tex"

cp "$ROOT_DIR/paper/ieee_paper.tex" "$MAIN_BUNDLE_DIR/ieee_paper.tex"
cp "$ROOT_DIR/paper/references.bib" "$MAIN_BUNDLE_DIR/references.bib"
cp "$ROOT_DIR/paper/ieee_paper.pdf" "$MAIN_BUNDLE_DIR/ieee_paper.pdf"
cp "$ROOT_DIR/architecture.png" "$MAIN_BUNDLE_DIR/architecture.png"
cp "$ROOT_DIR/paper/deployment_architecture.png" "$MAIN_BUNDLE_DIR/deployment_architecture.png"
cp "$ROOT_DIR/results/official_unsw_roc_curve.png" "$MAIN_BUNDLE_DIR/official_unsw_roc_curve.png"
cp "$ROOT_DIR/results/official_unsw_latency_under_load.png" "$MAIN_BUNDLE_DIR/official_unsw_latency_under_load.png"
cp "$ROOT_DIR/results/transfer_unsw_nsl_to_cicids_drift_timeline.png" "$MAIN_BUNDLE_DIR/transfer_unsw_nsl_to_cicids_drift_timeline.png"
cp "$ROOT_DIR/results/realtime_service_case_study.png" "$MAIN_BUNDLE_DIR/realtime_service_case_study.png"

perl -0pi -e 's!\.\./architecture\.png!architecture.png!g; s!deployment_architecture\.png!deployment_architecture.png!g; s!\.\./results/official_unsw_roc_curve\.png!official_unsw_roc_curve.png!g; s!\.\./results/official_unsw_latency_under_load\.png!official_unsw_latency_under_load.png!g; s!\.\./results/transfer_unsw_nsl_to_cicids_drift_timeline\.png!transfer_unsw_nsl_to_cicids_drift_timeline.png!g; s!\.\./results/realtime_service_case_study\.png!realtime_service_case_study.png!g' "$MAIN_BUNDLE_DIR/ieee_paper.tex"

cp "$ROOT_DIR/results/transfer_unsw_nsl_to_cicids_model_comparison.md" "$SUPPLEMENTARY_DIR/"
cp "$ROOT_DIR/results/drift_detector_study.md" "$SUPPLEMENTARY_DIR/"
cp "$ROOT_DIR/results/transfer_unsw_nsl_to_cicids_failure_case_analysis.md" "$SUPPLEMENTARY_DIR/"
cp "$ROOT_DIR/results/realtime_service_case_study.md" "$SUPPLEMENTARY_DIR/"
cp "$ROOT_DIR/results/transfer_unsw_nsl_to_cicids_realtime_streaming_evaluation.md" "$SUPPLEMENTARY_DIR/"
cp "$ROOT_DIR/results/advanced_experiment_summary.json" "$SUPPLEMENTARY_DIR/"
cp "$ROOT_DIR/results/transfer_unsw_nsl_to_cicids_full_summary.json" "$SUPPLEMENTARY_DIR/"
cp "$ROOT_DIR/results/realtime_service_case_study.json" "$SUPPLEMENTARY_DIR/"

rm -f "$SUBMISSION_DIR/drift_adaptive_intrusion_detection_main_manuscript.zip"
rm -f "$SUBMISSION_DIR/drift_adaptive_intrusion_detection_supplementary_review.zip"

(
  cd "$SUBMISSION_DIR"
  zip -rq drift_adaptive_intrusion_detection_main_manuscript.zip main_manuscript_bundle
  zip -rq drift_adaptive_intrusion_detection_supplementary_review.zip supplementary_review
)

echo "Built submission assets in $SUBMISSION_DIR"
