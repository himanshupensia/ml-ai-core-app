# Use a lightweight Python base image
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server script and the trained model
COPY server.py .
COPY model.pkl .

# Expose the port defined in server.py
EXPOSE 9001

# Command to run the Flask app
CMD ["python", "server.py"]