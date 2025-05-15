from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import base64
from io import BytesIO

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/", methods=["GET"])
def home():
    return "Masal API çalışıyor."

@app.route("/run/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"][0]
        base64_str = data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({"data": [caption]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
