from flask import Flask, request, jsonify
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import base64
from io import BytesIO
import os

app = Flask(__name__)

# Model ve yardımcılar
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cpu")
model.to(device)

@app.route("/")
def home():
    return "✅ Masal API Çalışıyor (vit-gpt2)"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"][0]
        base64_str = data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({"data": [caption]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
