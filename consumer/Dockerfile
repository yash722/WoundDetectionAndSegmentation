FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Upgrade pip
RUN pip install --upgrade pip

# Install necessary packages
RUN pip install numpy flask kafka-python pillow torchvision flask-socketio eventlet pillow

# Set the working directory inside the container
WORKDIR /app
COPY . /app

# Copy the wait-for-it script
COPY wait.sh /wait.sh
RUN chmod +x /wait.sh

# Copy the Flask app and templates
COPY flask_consumer.py /app/

# Expose port 5000 for Flask
EXPOSE 5001

# Run the Flask app only after Kafka is available
CMD ["/wait.sh", "kafka:9092", "--", "python", "flask_consumer.py"]

