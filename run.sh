#!/usr/bin/env bash
# Optical Lithography Simulator launcher
# Usage: ./run.sh [gui|sim|batch|parse] [args...]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

MODE=${1:-gui}
shift

case "$MODE" in
  gui)
    echo "Starting Optical Simulator GUI..."
    python3 "$SCRIPT_DIR/gui/main_window.py" "$@"
    ;;
  sim)
    echo "Running simulation..."
    python3 -c "
from pipeline.simulation_pipeline import SimulationPipeline
import sys
pipeline = SimulationPipeline()
result = pipeline.run_from_args(sys.argv[1:])
" "$@"
    ;;
  batch)
    echo "Running batch simulation..."
    python3 "$SCRIPT_DIR/pipeline/batch_runner.py" "$@"
    ;;
  parse)
    echo "Parsing PDF..."
    python3 -c "
from fileio.pdf_parser import parse_pdf
import sys, json
result = parse_pdf(sys.argv[1])
print(json.dumps(result.get('parameters', {}), indent=2))
" "$@"
    ;;
  *)
    echo "Usage: $0 [gui|sim|batch|parse] [args...]"
    exit 1
    ;;
esac
