import eventlet
eventlet.monkey_patch() 

from flask import Flask, render_template
from kafka import KafkaConsumer
from flask_socketio import SocketIO
from PIL import Image, ImageEnhance
import io
import base64
import numpy as np
import torch
import torchvision.transforms as transforms
import logging

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')  # Use eventlet for WebSocket management
logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
trained_deeplab = torch.load("model_trained_deeplab_resnet101.pt", map_location=torch.device(device))

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'wound_images',
    bootstrap_servers='kafka:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='wound_image_group',
    max_poll_interval_ms=300000,
    session_timeout_ms=60000,
    heartbeat_interval_ms=10000 
)

# Taking Segments out of the image using a sliding window (chosen 512 X 512 since the model is trained on those resolution of images)
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def pad_image(image, window_size):
    pad_height = (window_size[0] - image.shape[0] % window_size[0]) % window_size[0]
    pad_width = (window_size[1] - image.shape[1] % window_size[1]) % window_size[1]
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return padded_image

def segment_image(model, image, window_size=(512, 512), step_size=256):
    model.eval()
    image_np = np.array(image)
    image_np_padded = pad_image(image_np, window_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Transforms taken from the training dataset to prevent data leakage
        transforms.Normalize(mean=[0.4843, 0.3917, 0.3575], std=[0.2620, 0.2456, 0.2405])
    ])
    segmented_image = np.zeros((image_np_padded.shape[0], image_np_padded.shape[1]), dtype=np.uint8)
    for (x, y, patch) in sliding_window(image_np_padded, step_size, window_size):
        # Taking patches of image and using the segmentation model to get binary mask
        patch = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(patch)['out'].cpu().numpy().squeeze()
            # output = (output >= 252).astype(np.uint8)
        
        # Stiching the patches together
        segmented_image[y:y + window_size[1], x:x + window_size[0]] = output
    return segmented_image[:image_np.shape[0], :image_np.shape[1]]

def overlay_mask_onto_image(mask, image):
    mask_rgb = np.zeros_like(np.array(image))
    # Creating a red mask over the wound
    mask_rgb[:, :, 0] = (mask * 255)
    mask_pil = Image.fromarray(mask_rgb)
    # Applying the mask over the wound with high brightness
    mask_pil = ImageEnhance.Brightness(mask_pil).enhance(2)
    overlay = Image.blend(image, mask_pil, alpha=0.5)
    return overlay

# Function to process the image
def process_image(image):
    window_size = (512, 512)
    step_size = 256
    segmented_image = segment_image(trained_deeplab, image, window_size, step_size)
    overlay_image_mask = overlay_mask_onto_image(segmented_image, image)
    buffer = io.BytesIO()
    overlay_image_mask.save(buffer, format="PNG")
    buffer.seek(0)
    encoded_segmented_image = base64.b64encode(buffer.read()).decode('utf-8')
    return encoded_segmented_image

# Function to consume Kafka messages and emit them via WebSockets
def consume_messages():
    with app.app_context():
        for message in consumer:
            img_base64 = message.value.decode('utf-8')
            img_data = base64.b64decode(img_base64)
            img_buffer = io.BytesIO(img_data)
            image = Image.open(img_buffer).convert("RGB")
            print("Image successfully decoded and loaded")
            processed_image = process_image(image)
            socketio.emit('new_image', {'image': processed_image})
            logging.info("Image successfully decoded and loaded")

def start_consumer_task():
    socketio.start_background_task(target=consume_messages)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    start_consumer_task()
    socketio.run(app, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
