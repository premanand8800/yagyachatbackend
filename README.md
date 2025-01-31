# Rishi Backend

A sophisticated input validation and analysis system built with FastAPI, LangGraph, and NeMo Guardrails.

## Features

- Advanced input validation with NeMo Guardrails integration
- User preference management
- Rate limiting and authentication
- LLM-powered analysis
- Structured workflow using LangGraph

## Project Structure

```
rishi-backend/
├── app/
│   ├── models/         # Pydantic models
│   ├── middleware/     # Auth and rate limiting
│   ├── nodes/         # LangGraph nodes
│   ├── utils/         # Utility functions
│   └── config/        # Configuration
├── tests/            # Unit tests
├── .env              # Environment variables
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in `.env`:
   ```
   GROQ_API_KEY=your_groq_api_key
   JWT_SECRET=your_jwt_secret
   ```

## Running the Application

Start the server:
```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Health check endpoint
- `POST /api/process_input`: Process and validate user input

## Authentication

The API uses JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer your_jwt_token
```

## Rate Limiting

Endpoints are rate-limited to prevent abuse. Current limit: 5 requests per minute.

## Development

1. Make sure to run tests before submitting changes:
   ```bash
   pytest
   ```
2. Follow PEP 8 style guidelines
3. Update documentation as needed

## License

MIT License
