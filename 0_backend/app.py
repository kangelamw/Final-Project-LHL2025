from flask import Flask, request, jsonify, send_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
import base64
matplotlib.use('Agg')

import io
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model
model_name = "kangelamw/RoBERTa-political-bias-classifier-softmax"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device=torch.device("cuda")
model.to(device)
model.eval()

# Labels
LABELS = ["Liberal", "Neutral", "Conservative"]

# Setup Flask app
app = Flask(__name__)

# Inference function
def classify_text(texts):
    inputs = tokenizer(texts, 
                       return_tensors="pt", 
                       truncation=True, 
                       padding=True)
    
    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    logits = logits.to(device)
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1).tolist()

    # Return label + confidence score
    results = []
    for i in range(probs.size(0)):
        # Convert tensor to list and multiply by 100 for percentage
        text_probs = probs[i].tolist()
        percentages = {LABELS[j]: round(text_probs[j] * 100, 2) for j in range(len(LABELS))}
        results.append(percentages)
    
    return results, probs

# Spider chart generation
def create_spider_chart(probs):
    categories = LABELS
    num_vars = len(categories)

    # Set the angle for each axis
    angles = np.linspace(0, 2 * np.pi, 
                         num_vars, 
                         endpoint=False).tolist()

    # Make the plot circular
    probs = probs.tolist() + probs.tolist()[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), 
                           dpi=150, 
                           subplot_kw=dict(polar=True))
    ax.fill(angles, 
            probs, 
            color='blue', 
            alpha=0.5)
    ax.plot(angles, 
            probs, 
            color='blue', 
            linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf
  
# API Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        texts = data.get("text", None)
        if not texts:
            return jsonify({"error": "No text provided"}), 400

        if isinstance(texts, str):
            texts = [texts]

        predictions, probs = classify_text(texts)
        buf = create_spider_chart(probs[0])
        
        # Encode image as base64 string
        chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Return with predictions and the chart
        return jsonify({
            "predictions": predictions,
            "spider_chart": chart_base64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/test", methods=["GET"])
def test():
    return "Hello I'm alive.", 200

# print("Available Routes:", app.url_map)

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000, debug=True)