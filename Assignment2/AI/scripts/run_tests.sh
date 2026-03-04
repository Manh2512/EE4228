#!/usr/bin/env bash
# run_tests.sh — Run the full test suite for the Face Recognition System.
#
# Usage:
#   chmod +x scripts/run_tests.sh
#   ./scripts/run_tests.sh
#
# Environment variables:
#   SKIP_INTEGRATION=1   Skip tests that require downloaded model weights.
#   VERBOSE=1            Show stdout/stderr from each test.

set -euo pipefail

# ── Resolve project root (directory containing this script's parent) ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo " Face Recognition System — Test Suite"
echo " Working directory: ${PROJECT_ROOT}"
echo "============================================================"

# ── Check Python ──────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" &>/dev/null; then
    echo "[ERROR] Python interpreter not found: ${PYTHON}"
    echo "        Set the PYTHON env var to point to a valid interpreter."
    exit 1
fi
echo "Python: $("${PYTHON}" --version)"

# ── Check pytest ──────────────────────────────────────────────────────────────
if ! "${PYTHON}" -m pytest --version &>/dev/null; then
    echo "[INFO]  pytest not found. Installing …"
    "${PYTHON}" -m pip install pytest --quiet
fi
echo "pytest: $("${PYTHON}" -m pytest --version)"
echo ""

# ── Build pytest arguments ────────────────────────────────────────────────────
PYTEST_ARGS=(
    "tests/"
    "--tb=short"
    "--color=yes"
    "-q"
)

if [[ "${VERBOSE:-0}" == "1" ]]; then
    PYTEST_ARGS+=("-v" "-s")
fi

if [[ "${SKIP_INTEGRATION:-0}" == "1" ]]; then
    # Skip tests marked as requiring model files (they use pytest.mark.skipif
    # internally, but we can also add an explicit -k filter here).
    echo "[INFO]  SKIP_INTEGRATION=1: integration tests will be skipped if model files absent."
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Running: ${PYTHON} -m pytest ${PYTEST_ARGS[*]}"
echo "------------------------------------------------------------"

"${PYTHON}" -m pytest "${PYTEST_ARGS[@]}"
EXIT_CODE=$?

echo "------------------------------------------------------------"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo " All tests passed."
else
    echo " Some tests FAILED (exit code ${EXIT_CODE})."
fi
echo "============================================================"
exit ${EXIT_CODE}
