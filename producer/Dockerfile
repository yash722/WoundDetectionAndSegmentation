FROM python:3.9-slim

# Install necessary packages
RUN pip install flask kafka-python pillow numpy

# Set the working directory inside the container
WORKDIR /app
COPY . /app

# Copy the wait-for-it script
COPY wait.sh /wait.sh
RUN chmod +x /wait.sh

# Copy the Flask app and templates
COPY flask_producer.py /app/
COPY templates/ /app/templates/

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app only after Kafka is available
CMD ["/wait.sh", "kafka:9092", "--", "python", "flask_producer.py"]