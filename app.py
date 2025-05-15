from flask import Flask, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from io import BytesIO
import base64

app = Flask(__name__)

# Hafif model: BLIP base
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/")
def home():
    return "✅ Masal API ayakta."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"][0]
        base64_str = data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return jsonify({"data": [caption]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Render için dış dünyaya açık port
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
