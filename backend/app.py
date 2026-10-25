from pathlib import Path
import joblib, pandas as pd
from flask import Flask, jsonify, request

BUNDLE = Path(__file__).parent / "models" / "model_bundle.joblib"
bundle = joblib.load(BUNDLE)
pipe, schema = bundle["pipeline"], bundle["schema"]
FEATURES = schema["features"]

app = Flask(__name__)

@app.after_request
def cors(resp):  # allow Angular dev server / pages to call locally
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.get("/health")
def health():
    return jsonify(status="ok", n_features=len(FEATURES), classes=schema["classes"])

@app.get("/schema")
def get_schema():
    return jsonify(schema)

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="JSON object body required"), 400
    missing = [f for f in FEATURES if f not in data]
    extra = [k for k in data if k not in FEATURES]
    if missing or extra:
        return jsonify(error="schema mismatch", missing=missing, unexpected=extra), 400
    try:
        row = pd.DataFrame([{f: float(data[f]) for f in FEATURES}])
    except (TypeError, ValueError):
        return jsonify(error="all feature values must be numeric"), 400
    proba = pipe.predict_proba(row)[0]
    i = int(proba.argmax())
    return jsonify(prediction=str(pipe.classes_[i]), confidence=round(float(proba[i]), 4),
                   is_attack=str(pipe.classes_[i]) != "BENIGN",
                   probabilities={str(c): round(float(p), 4) for c, p in zip(pipe.classes_, proba)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
