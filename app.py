from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import base64
from io import BytesIO
import os

# Model ve tokenizer yükle
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Flask uygulaması
app = Flask(__name__)

@app.route("/run/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"][0]  # base64 img
        base64_str = data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return jsonify({"data": [caption]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# PORT bind
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
