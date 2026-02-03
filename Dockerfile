FROM python:3.9-slim

WORKDIR /app

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/app.py .
COPY inference/model.joblib .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
