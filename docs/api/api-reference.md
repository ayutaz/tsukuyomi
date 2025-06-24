# Tsukuyomi TTS API Reference

## Overview

The Tsukuyomi TTS API provides a comprehensive interface for text-to-speech synthesis with advanced features including emotion control, style transfer, and voice morphing.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

API requests require an API key to be included in the header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/synthesize
```

## Endpoints

### 1. Synthesize Speech

Generate speech from text with various customization options.

#### POST `/synthesize`

##### Request Body

```json
{
  "text": "string",
  "speaker_id": "integer",
  "language": "string",
  "emotion": "string",
  "emotion_intensity": "float",
  "style": "string",
  "speed": "float",
  "pitch_shift": "float",
  "energy": "float",
  "output_format": "string"
}
```

##### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize (max 5000 characters) |
| `speaker_id` | integer | No | 0 | Speaker ID (0-499) |
| `language` | string | No | "ja" | Language code (ja, en, zh, ko) |
| `emotion` | string | No | "neutral" | Emotion type (see [Emotion Types](#emotion-types)) |
| `emotion_intensity` | float | No | 1.0 | Emotion intensity (0.0-2.0) |
| `style` | string | No | "normal" | Speaking style (see [Style Types](#style-types)) |
| `speed` | float | No | 1.0 | Speaking speed (0.5-2.0) |
| `pitch_shift` | float | No | 0.0 | Pitch shift in semitones (-12.0 to 12.0) |
| `energy` | float | No | 1.0 | Energy/volume scale (0.5-2.0) |
| `output_format` | string | No | "wav" | Output format (wav, mp3, ogg) |

##### Response

```json
{
  "audio_url": "string",
  "audio_base64": "string",
  "duration": "float",
  "sample_rate": "integer",
  "speaker_name": "string",
  "synthesis_time": "float"
}
```

##### Example

```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "こんにちは、私は月読です。",
    "speaker_id": 0,
    "emotion": "happy",
    "emotion_intensity": 1.5
  }'
```

### 2. Batch Synthesis

Generate multiple audio files in a single request.

#### POST `/synthesize/batch`

##### Request Body

```json
{
  "requests": [
    {
      "id": "string",
      "text": "string",
      "speaker_id": "integer",
      "emotion": "string"
    }
  ],
  "common_settings": {
    "output_format": "string",
    "sample_rate": "integer"
  }
}
```

##### Response

```json
{
  "results": [
    {
      "id": "string",
      "status": "string",
      "audio_url": "string",
      "error": "string"
    }
  ],
  "total_time": "float"
}
```

### 3. Style Transfer

Apply style from a reference audio to synthesized speech.

#### POST `/style-transfer`

##### Request Body (multipart/form-data)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `reference_audio` | file | Yes | Reference audio file (WAV, MP3) |
| `speaker_id` | integer | No | Target speaker ID |
| `transfer_strength` | float | No | Style transfer strength (0.0-1.0) |

##### Response

```json
{
  "audio_url": "string",
  "style_embedding": "array",
  "similarity_score": "float"
}
```

### 4. Voice Morphing

Morph between multiple speakers.

#### POST `/voice-morph`

##### Request Body

```json
{
  "text": "string",
  "speakers": [
    {
      "id": "integer",
      "weight": "float"
    }
  ],
  "morph_type": "string"
}
```

##### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `speakers` | array | Yes | Array of speaker IDs and weights |
| `morph_type` | string | No | Morphing algorithm ("linear", "spherical") |

### 5. Streaming Synthesis

Real-time streaming synthesis with WebSocket.

#### WebSocket `/ws/synthesize/stream`

##### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/synthesize/stream');
```

##### Message Format

Send:
```json
{
  "text": "string",
  "speaker_id": "integer",
  "chunk_size": "integer"
}
```

Receive:
```json
{
  "type": "audio_chunk",
  "data": "base64_string",
  "chunk_id": "integer",
  "is_final": "boolean"
}
```

### 6. Speaker Management

#### GET `/speakers`

Get list of available speakers.

##### Response

```json
{
  "speakers": [
    {
      "id": "integer",
      "name": "string",
      "language": "string",
      "gender": "string",
      "age_group": "string",
      "description": "string",
      "preview_url": "string"
    }
  ],
  "total": "integer"
}
```

#### GET `/speakers/{speaker_id}`

Get detailed information about a specific speaker.

### 7. Model Information

#### GET `/models`

Get information about available models.

##### Response

```json
{
  "models": [
    {
      "id": "string",
      "name": "string",
      "version": "string",
      "languages": ["string"],
      "features": ["string"],
      "size_mb": "integer"
    }
  ]
}
```

### 8. Health Check

#### GET `/health`

Check API health status.

##### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "integer",
  "gpu_available": "boolean",
  "models_loaded": ["string"]
}
```

## Data Types

### Emotion Types

- `neutral` - Neutral emotion
- `happy` - Happy/joyful
- `sad` - Sad/melancholic
- `angry` - Angry/irritated
- `fearful` - Fearful/anxious
- `surprised` - Surprised/shocked
- `disgusted` - Disgusted
- `excited` - Excited/enthusiastic
- `calm` - Calm/relaxed
- `confident` - Confident/assured

### Style Types

- `normal` - Normal speaking style
- `casual` - Casual/informal
- `formal` - Formal/polite
- `energetic` - Energetic/dynamic
- `gentle` - Gentle/soft
- `serious` - Serious/grave
- `cheerful` - Cheerful/bright
- `dramatic` - Dramatic/theatrical
- `whisper` - Whispering
- `shouting` - Shouting/loud

## Error Responses

All endpoints return errors in the following format:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `FORBIDDEN` | 403 | Access forbidden |
| `NOT_FOUND` | 404 | Resource not found |
| `TEXT_TOO_LONG` | 400 | Text exceeds maximum length |
| `INVALID_SPEAKER` | 400 | Invalid speaker ID |
| `INVALID_EMOTION` | 400 | Invalid emotion type |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |

## Rate Limiting

- Default: 100 requests per minute
- Batch synthesis: 10 requests per minute
- Streaming: 5 concurrent connections

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## SDKs and Client Libraries

### Python

```python
from tsukuyomi import TsukuyomiClient

client = TsukuyomiClient(api_key="your-api-key")
audio = client.synthesize(
    text="こんにちは",
    speaker_id=0,
    emotion="happy"
)
```

### JavaScript/TypeScript

```typescript
import { TsukuyomiClient } from '@tsukuyomi/client';

const client = new TsukuyomiClient({ apiKey: 'your-api-key' });
const audio = await client.synthesize({
  text: 'こんにちは',
  speakerId: 0,
  emotion: 'happy'
});
```

### cURL Examples

```bash
# Basic synthesis
curl -X POST http://localhost:8000/api/v1/synthesize \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "speaker_id": 0}'

# With emotion and style
curl -X POST http://localhost:8000/api/v1/synthesize \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "感情豊かな音声です",
    "speaker_id": 5,
    "emotion": "excited",
    "emotion_intensity": 1.8,
    "style": "energetic",
    "speed": 1.2
  }'

# Style transfer
curl -X POST http://localhost:8000/api/v1/style-transfer \
  -H "X-API-Key: your-api-key" \
  -F "text=スタイル転送のテストです" \
  -F "reference_audio=@reference.wav" \
  -F "transfer_strength=0.7"
```

## Changelog

### v1.0.0 (2024-01-01)
- Initial release
- Basic synthesis API
- Emotion control
- Style transfer
- Voice morphing

### v1.1.0 (2024-02-01)
- Added streaming synthesis
- Batch processing improvements
- New emotion types
- Performance optimizations