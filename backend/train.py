"""Train RF on the paper's 22 Information-Gain features (weight > 0.4, Table 6)."""
import json, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "Wednesday-workingHours_pcap_ISCX.csv"

# Paper Table 6, group ">0.4" (22 features), names from Table 5
FEATURES = [
 "Packet Length Std","Total Length of Bwd Packets","Subflow Bwd Bytes","Destination Port",
 "Packet Length Variance","Bwd Packet Length Mean","Avg Bwd Segment Size","Bwd Packet Length Max",
 "Init_Win_bytes_backward","Total Length of Fwd Packets","Subflow Fwd Bytes","Init_Win_bytes_forward",
 "Average Packet Size","Packet Length Mean","Max Packet Length","Fwd Packet Length Max",
 "Flow IAT Max","Bwd Header Length","Flow Duration","Fwd IAT Max","Fwd Header Length","Fwd IAT Total"]

df = pd.read_csv(CSV, skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["Label"])
# paper: use 20% of data, then 70/30 split
parts = [g.sample(frac=0.2, random_state=42) if len(g) > 100 else g for _, g in df.groupby("Label")]
df = pd.concat(parts)
X, y = df[FEATURES].astype(float), df["Label"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("rf", RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=42, min_samples_leaf=2))])
t = time.time(); pipe.fit(Xtr, ytr); print(f"train {time.time()-t:.1f}s")
pred = pipe.predict(Xte)
acc = accuracy_score(yte, pred)
print(f"accuracy {acc:.4f}  (paper RF/22 feats: 0.9986)")
print(classification_report(yte, pred, zero_division=0))

schema = {"features": FEATURES, "classes": [str(c) for c in pipe.classes_],
          "ranges": {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES}}
out = ROOT / "backend" / "models" / "model_bundle.joblib"
joblib.dump({"pipeline": pipe, "schema": schema, "test_accuracy": float(acc)}, out, compress=3)
print("saved", out, f"{out.stat().st_size/1e6:.1f} MB")

row = Xte[yte == "DoS Hulk"].iloc[[0]]
row.to_csv(ROOT / "data" / "sample_inputs.csv", index=False)
json.dump({k: float(v) for k, v in row.iloc[0].items()}, open(ROOT / "backend" / "test_payload.json", "w"), indent=2)
json.dump(schema, open(ROOT / "backend" / "models" / "schema.json", "w"), indent=2)
