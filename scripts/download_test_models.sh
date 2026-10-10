#!/usr/bin/env bash
# Download test GGUF models for EIF-Runtime test suite
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/tests/data"

mkdir -p "${DATA_DIR}"

echo "=========================================================="
echo "  EIF-Runtime: Downloading Test GGUF Models"
echo "=========================================================="

# 1. Jina Embeddings v2 Base Code (ALiBi Model)
# Used for ALiBi verification and benchmark
JINA_TARGET="${DATA_DIR}/jina-embeddings-v2-base-code.gguf"
JINA_URL="https://huggingface.co/second-state/jina-embeddings-v2-base-code-GGUF/resolve/main/jina-embeddings-v2-base-code-Q8_0.gguf"

if [ -f "${JINA_TARGET}" ]; then
    echo "✓ Jina ALiBi model already present: ${JINA_TARGET}"
elif [ -f "${DATA_DIR}/test_jina_alibi.gguf" ]; then
    echo "✓ Local ALiBi test model already present: ${DATA_DIR}/test_jina_alibi.gguf"
else
    echo "⬇ Downloading Jina Embeddings v2 (ALiBi Q8_0)..."
    if curl -L --fail --progress-bar -o "${JINA_TARGET}.tmp" "${JINA_URL}"; then
        mv "${JINA_TARGET}.tmp" "${JINA_TARGET}"
        echo "✓ Successfully downloaded: ${JINA_TARGET}"
    else
        echo "⚠ Warning: Could not download from ${JINA_URL}."
        echo "  If offline, tests will skip live ALiBi downloads."
        rm -f "${JINA_TARGET}.tmp"
    fi
fi

# 2. MiniLM / Granite 30M class Embedding Model
MINILM_TARGET="${DATA_DIR}/minilm.gguf"
MINILM_URL="https://huggingface.co/second-state/all-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf"

if [ -f "${MINILM_TARGET}" ]; then
    echo "✓ MiniLM model already present: ${MINILM_TARGET}"
elif [ -f "${ROOT_DIR}/../../models/dist/minilm.gguf" ]; then
    echo "✓ Found local workspace model: ${ROOT_DIR}/../../models/dist/minilm.gguf"
else
    echo "⬇ Downloading MiniLM Q8_0..."
    if curl -L --fail --progress-bar -o "${MINILM_TARGET}.tmp" "${MINILM_URL}"; then
        mv "${MINILM_TARGET}.tmp" "${MINILM_TARGET}"
        echo "✓ Successfully downloaded: ${MINILM_TARGET}"
    else
        echo "⚠ Warning: Could not download MiniLM."
        rm -f "${MINILM_TARGET}.tmp"
    fi
fi

echo "All test models setup complete."
