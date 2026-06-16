import os
import sys

from flask import Flask, jsonify, render_template, request


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


app = Flask(__name__)

predict_email = None
model_load_error = None

try:
    from backend.predict import MODEL_KIND, labels, metrics, predict_email
except Exception as error:
    MODEL_KIND = None
    labels = []
    metrics = {}
    model_load_error = str(error)
    print("Model load error:", model_load_error)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "ok": predict_email is not None,
        "model_kind": MODEL_KIND,
        "labels": labels,
        "metrics": metrics,
        "error": model_load_error,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if predict_email is None:
            return jsonify({
                "error": "Model not loaded",
                "details": model_load_error,
            }), 500

        data = request.get_json()

        if not data or "email" not in data:
            return jsonify({"error": "No email text provided"}), 400

        return jsonify(predict_email(data["email"]))

    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

