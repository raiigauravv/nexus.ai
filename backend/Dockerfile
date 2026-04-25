FROM python:3.11-slim

WORKDIR /app

# Install required system dependencies for psycopg2 and ML packages
RUN apt-get update && \
    apt-get install -y libpq-dev gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app source code
COPY . .

# Copy pre-trained model artifacts (fraud XGBoost, recommend SVD)
RUN mkdir -p /app/models/fraud /app/data/fraud /app/models/recommend /app/data/recommend
COPY models/fraud/ /app/models/fraud/
COPY models/recommend/ /app/models/recommend/

# Expose the API port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
