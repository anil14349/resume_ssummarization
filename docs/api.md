# API Documentation

## Endpoints

### Generate Summary
```http
POST /generate-summary/
```

Generates a professional summary from a resume file.

**Parameters:**
- `file`: Resume file (DOCX only)
- `model_type`: Model to use (gpt2, t5, or bart)
- `parser_type`: Parser to use (ats or industry)
- `debug`: Enable debug logging (optional)

**Response:**
```json
{
    "filename": "example.docx",
    "model_type": "gpt2",
    "parser_type": "ats",
    "summary": "Generated professional summary..."
}
```

### List Models
```http
GET /models
```

Lists available models for summary generation.

**Response:**
```json
{
    "models": [
        {
            "id": "gpt2",
            "name": "GPT-2",
            "description": "OpenAI's GPT-2 model fine-tuned for resume summarization"
        },
        {
            "id": "t5",
            "name": "T5",
            "description": "Google's T5 model fine-tuned for resume summarization"
        },
        {
            "id": "bart",
            "name": "BART",
            "description": "Facebook's BART model fine-tuned for resume summarization"
        }
    ]
}
```

### List Parsers
```http
GET /parsers
```

Lists available resume parsers.

**Response:**
```json
{
    "parsers": [
        {
            "id": "ats",
            "name": "ATS Parser",
            "description": "Parser optimized for ATS formatted resumes"
        },
        {
            "id": "industry",
            "name": "Industry Parser",
            "description": "Parser optimized for industry-specific resume formats"
        }
    ]
}
```

### Health Check
```http
GET /health
```

Checks API health status.

**Response:**
```json
{
    "status": "healthy"
}
```

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request (invalid input)
- 500: Internal Server Error

Error responses include a detail message:
```json
{
    "detail": "Error message describing the problem"
}
```

## CORS

The API supports Cross-Origin Resource Sharing (CORS) with the following configuration:
- All origins allowed
- All methods allowed
- All headers allowed
- Credentials supported

## Rate Limiting

Currently, no rate limiting is implemented. Consider adding rate limiting for production deployment.
