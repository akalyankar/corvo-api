# Corvo API

Flask REST API for Corvo Vendor Normalization & Categorization System.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure `.env` file with your Corvo path and settings

3. Ensure Milvus is running:
```bash
cd ../corvo
docker-compose up -d
```

4. Run the API:
```bash
python run.py
```

## API Endpoints

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /api/v1/process` - Full pipeline (Normalize + Categorize)
- `POST /api/v1/normalize` - Normalization only
- `POST /api/v1/categorize/by-flag` - Categorize by flag
- `POST /api/v1/categorize/rules-only` - Rules-only categorization
- `POST /api/v1/categorize/ml` - ML categorization
- `POST /api/v1/categorize/llm` - LLM categorization
- `POST /api/v1/categorize/full` - Full cascading pipeline
- `POST /api/v1/normalize-and-categorize/rules-only` - Norm + Rules Cat
- `POST /api/v1/normalize-and-categorize/ml` - Norm + ML Cat
- `POST /api/v1/normalize-and-categorize/llm` - Norm + LLM Cat
- `GET /api/v1/taxonomy/schema` - Get taxonomy schema

## Documentation

API runs on http://localhost:8000
