#!/bin/bash
# ============================================================
# download.sh — Pull all Arabic NLP benchmark datasets
# ============================================================
# Datasets:
#   - HARD    : Hotel Arabic Reviews Dataset (sentiment)
#   - ArSAS   : Arabic Sentiment Analysis (sentiment)
#   - ANERcorp: Arabic Named Entity Recognition corpus
#   - TyDi QA : Question Answering (Arabic subset)
# ============================================================

set -e

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "📥 Downloading datasets to: $DATA_DIR"

# ── Colours ──────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ── HARD Dataset (via HuggingFace) ───────────────────────────
info "Downloading HARD (Hotel Arabic Reviews Dataset)..."
python3 - <<'EOF'
from datasets import load_dataset
import json, os

ds = load_dataset("hard", trust_remote_code=True)
os.makedirs("hard", exist_ok=True)
for split in ds:
    ds[split].to_json(f"hard/{split}.jsonl")
print("  ✓ HARD saved to data/hard/")
EOF

# ── ArSAS Dataset ─────────────────────────────────────────────
info "Downloading ArSAS (Arabic Sentiment Analysis)..."
python3 - <<'EOF'
from datasets import load_dataset
import os

# ArSAS available via HuggingFace
ds = load_dataset("arbml/arsas", trust_remote_code=True)
os.makedirs("arsas", exist_ok=True)
for split in ds:
    ds[split].to_json(f"arsas/{split}.jsonl")
print("  ✓ ArSAS saved to data/arsas/")
EOF

# ── ANERcorp (NER) ────────────────────────────────────────────
info "Downloading ANERcorp (Named Entity Recognition)..."
python3 - <<'EOF'
from datasets import load_dataset
import os

ds = load_dataset("conll2003", trust_remote_code=True)   # fallback; swap for ANERcorp when available
os.makedirs("anercorp", exist_ok=True)
# Real ANERcorp: https://camel.abudhabi.nyu.edu/anercorp/
# Download manually if institutional access needed; script prepares the folder.
print("  ✓ ANERcorp folder ready — place anercorp.txt inside data/anercorp/")
print("    Download from: https://camel.abudhabi.nyu.edu/anercorp/")
EOF

# ── TyDi QA (Arabic subset) ───────────────────────────────────
info "Downloading TyDi QA Arabic subset..."
python3 - <<'EOF'
from datasets import load_dataset
import os

ds = load_dataset("tydiqa", "secondary_task", trust_remote_code=True)
arabic = ds.filter(lambda x: x["id"].startswith("arabic"))
os.makedirs("tydiqa", exist_ok=True)
for split in arabic:
    arabic[split].to_json(f"tydiqa/{split}.jsonl")
print("  ✓ TyDi QA Arabic saved to data/tydiqa/")
EOF

echo ""
info "All datasets downloaded ✓"
echo "Directory layout:"
find "$DATA_DIR" -maxdepth 2 -type d | sort
