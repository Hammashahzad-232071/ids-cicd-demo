import json, joblib, pandas as pd
from pathlib import Path
d = Path(__file__).parent
b = joblib.load(d / "models" / "model_bundle.joblib")
row = pd.read_csv(d.parent / "data" / "sample_inputs.csv")[b["schema"]["features"]]
print("predict:", b["pipeline"].predict(row)[0], "(expected: DoS Hulk)")
