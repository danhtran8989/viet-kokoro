# API Reference

## Base URL

```
http://localhost:8000
```

## Endpoints

### List Voices

```
GET /voices
```

**Response:**

```json
{
  "voices": {
    "diem_trinh": "Diem Trinh",
    "hung_thinh": "Hung Thinh",
    "mai_linh": "Mai Linh"
  }
}
```

### Text-to-Speech (Audio)

```
POST /tts
Content-Type: application/json
```

**Request:**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | (required) | Vietnamese text to synthesize |
| `voice` | string | `diem_trinh` | Voice ID |
| `speed` | float | `1.0` | Speed multiplier (0.75 - 1.25) |

**Example:**

```json
{
  "text": "Xin chao ban",
  "voice": "mai_linh",
  "speed": 1.0
}
```

**Response:** WAV audio file (`audio/wav`)

### Text-to-Speech (JSON)

```
POST /tts/json
Content-Type: application/json
```

Same request as `/tts`.

**Response:**

```json
{
  "voice": "Mai Linh",
  "sample_rate": 24000,
  "duration_seconds": 1.23
}
```

## Error Responses

| Status | Meaning |
|---|---|
| 400 | Invalid request (empty text, unknown voice, phoneme too long) |
| 422 | Validation error (speed out of range) |
