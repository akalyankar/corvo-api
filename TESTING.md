# Testing Guide for Corvo API

## Prerequisites

1. **Python 3.8+** installed
2. **Milvus** running (via Docker Compose)
3. **Corvo system** properly set up with embedding model

## Setup Steps

### 1. Start Milvus (Required for API)

```bash
cd ../corvo
docker-compose up -d
```

Wait for Milvus to be ready (usually 30-60 seconds).

### 2. Configure Environment Variables

Create a `.env` file in `corvo-api/` directory:

```env
# Corvo path (relative to corvo-api directory)
CORVO_PATH=../corvo

# Flask settings
FLASK_HOST=0.0.0.0
FLASK_PORT=8000
FLASK_DEBUG=True

# Milvus configuration
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION=corvo

# CORS (allow all origins for development)
CORS_ORIGINS=*
```

### 3. Install Dependencies

```bash
cd corvo-api
pip install -r requirements.txt
```

### 4. Start the API Server

```bash
python run.py
```

You should see:
```
✓ Corvo service initialized successfully
 * Running on http://0.0.0.0:8000
```

## Testing Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "corvo-api",
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

### Readiness Check

```bash
curl http://localhost:8000/ready
```

Expected response:
```json
{
  "status": "ready",
  "embedder_loaded": true,
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

### Test Full Pipeline (Normalize + Categorize)

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "vendor_name": "Microsoft Corp",
        "gl_description": "Software license subscription",
        "amount": 1000.00
      },
      {
        "vendor_name": "MSFT",
        "gl_description": "Cloud services",
        "amount": 500.00
      }
    ],
    "vendor_column": "vendor_name",
    "use_ml": true,
    "use_llm": false
  }'
```

### Test Normalization Only

```bash
curl -X POST http://localhost:8000/api/v1/normalize \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"vendor_name": "Microsoft Corp"},
      {"vendor_name": "MSFT"},
      {"vendor_name": "Microsoft Corporation"}
    ],
    "vendor_column": "vendor_name"
  }'
```

### Test Categorization Only

```bash
curl -X POST http://localhost:8000/api/v1/categorize \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "vendor_name": "Microsoft",
        "gl_description": "Software license"
      }
    ],
    "vendor_column": "vendor_name",
    "description_column": "gl_description",
    "use_ml": true
  }'
```

### Get Taxonomy Schema

```bash
curl http://localhost:8000/api/v1/taxonomy/schema
```

## Common Issues

### Issue: "Corvo path does not exist"
**Solution**: Check that `CORVO_PATH` in `.env` points to the correct corvo directory.

### Issue: "Embedding model not found"
**Solution**: Ensure the embedding model file exists in the corvo directory (usually `models/Qwen3-Embedding-4B-Q8_0.gguf`).

### Issue: "Failed to connect to Milvus"
**Solution**: 
1. Check that Milvus is running: `docker ps`
2. Verify `MILVUS_URI` in `.env` matches your Milvus instance
3. Wait a bit longer for Milvus to fully start

### Issue: CORS errors from frontend
**Solution**: Ensure `CORS_ORIGINS` in `.env` includes your frontend URL or use `*` for development.

