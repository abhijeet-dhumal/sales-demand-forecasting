#!/bin/bash
# Render the canonical Mermaid sources into the PNGs embedded in the notebooks.
# Usage: ./generate-diagrams.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if ! command -v mmdc >/dev/null 2>&1; then
    echo "Mermaid CLI (mmdc) is required. Install it with: npm install --global @mermaid-js/mermaid-cli" >&2
    exit 1
fi

for input in "$SCRIPT_DIR"/01-features-workflow.mmd "$SCRIPT_DIR"/02-training-workflow.mmd "$SCRIPT_DIR"/03-inference-workflow.mmd "$SCRIPT_DIR"/sequence-diagram.mmd; do
    output="$SCRIPT_DIR/../$(basename "${input%.mmd}").png"
    mmdc --input "$input" --output "$output" --backgroundColor white
    echo "Rendered $(basename "$output")"
done
