from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_required_files():
    for p in ["backend/app.py", "backend/models/model_bundle.joblib", "backend/test_payload.json",
              "data/sample_inputs.csv", ".github/workflows/pages.yml", ".gitignore",
              "frontend/proxy.conf.json", "frontend/package.json"]:
        assert (ROOT / p).exists(), f"missing {p}"

def test_gitignore_entries():
    txt = (ROOT / ".gitignore").read_text()
    for e in [".venv/", "frontend/node_modules/", "frontend/dist/", "__pycache__/", ".angular/"]:
        assert e in txt
