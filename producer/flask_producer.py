from flask import Flask, request, redirect, url_for, render_template
from kafka import KafkaProducer
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)


def encode_image(image_bytes):
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return encoded

# Initialize the Kafka producer
producer = KafkaProducer(bootstrap_servers='kafka:9092')

# Route to upload image
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    digital_image = request.files['image']
    image = Image.open(digital_image.stream).convert('RGB')
    if 'image' not in request.files:
        return "No file part", 400

    if digital_image:
        # Read the image and convert to RGB using Pillow
        image = Image.open(digital_image.stream).convert('RGB')
        # Convert image to bytes and encode in Base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # Send the image to Kafka
        producer.send('wound_images', value=img_base64.encode('utf-8'))
        producer.flush()
        app.logger.info(f"Image sent to Kafka: {digital_image.filename}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
