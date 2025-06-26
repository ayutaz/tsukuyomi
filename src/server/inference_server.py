"""
Tsukuyomi Inference Server - Production-ready TTS API server

Features:
- RESTful API and WebSocket support
- Batch processing
- Caching
- Rate limiting
- Health monitoring
"""

import asyncio
import hashlib
import io
import logging
import time
from collections import deque
from typing import List, Optional

import numpy as np
import redis
import soundfile as sf
import torch
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..tsukuyomi_tts import TsukuyomiTTS

logger = logging.getLogger(__name__)


# Request/Response models
class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0
    emotion: str = "neutral"
    style: str = "normal"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    pitch_shift: float = Field(0.0, ge=-12, le=12)
    energy: float = Field(1.0, ge=0.5, le=2.0)
    output_format: str = "wav"  # wav, mp3, ogg
    streaming: bool = False


class MorphRequest(BaseModel):
    text: str
    speaker_ids: List[int]
    speaker_weights: List[float]
    emotion_ids: Optional[List[str]] = None
    emotion_weights: Optional[List[float]] = None
    speed: float = Field(1.0, ge=0.5, le=2.0)
    output_format: str = "wav"


class BatchTTSRequest(BaseModel):
    requests: List[TTSRequest]
    parallel: bool = True


class VoiceCloneRequest(BaseModel):
    text: str
    emotion: str = "neutral"
    output_format: str = "wav"


# Server configuration
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_text_length: int = 1000
    max_batch_size: int = 10
    enable_gpu: bool = True
    num_workers: int = 4
    rate_limit: int = 100  # requests per minute
    cors_origins: List[str] = ["*"]


class InferenceServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI(title="Tsukuyomi TTS API", version="1.0.0")

        # Initialize TTS system
        device = "cuda" if config.enable_gpu and torch.cuda.is_available() else "cpu"
        self.tts = TsukuyomiTTS(device=device, checkpoint_dir=config.model_path)

        # Initialize cache
        if config.cache_enabled:
            self.cache = redis.Redis(host="localhost", port=6379, decode_responses=True)
        else:
            self.cache = None

        # Rate limiting
        self.request_history = deque(maxlen=config.rate_limit * 10)

        # Setup routes
        self._setup_routes()

        # Setup middleware
        self._setup_middleware()

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            return {
                "name": "Tsukuyomi TTS API",
                "version": "1.0.0",
                "status": "running",
            }

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "device": self.tts.device,
                "models_loaded": True,
                "metrics": self.metrics,
            }

        @self.app.post("/synthesize")
        async def synthesize(request: TTSRequest):
            """Synthesize speech from text."""
            try:
                # Check rate limit
                if not self._check_rate_limit():
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

                # Validate request
                if len(request.text) > self.config.max_text_length:
                    raise HTTPException(status_code=400, detail="Text too long")

                # Check cache
                cache_key = self._get_cache_key(request)
                if self.cache and not request.streaming:
                    cached = self._get_cached(cache_key)
                    if cached:
                        self.metrics["cache_hits"] += 1
                        return StreamingResponse(
                            io.BytesIO(cached),
                            media_type=f"audio/{request.output_format}",
                        )

                self.metrics["cache_misses"] += 1

                # Generate audio
                start_time = time.time()
                audio = await self._generate_audio(request)
                latency = time.time() - start_time

                # Update metrics
                self._update_metrics(success=True, latency=latency)

                # Convert to requested format
                audio_bytes = self._convert_audio(audio, request.output_format)

                # Cache result
                if self.cache and not request.streaming:
                    self._cache_result(cache_key, audio_bytes)

                # Return response
                if request.streaming:
                    return StreamingResponse(
                        self._stream_audio(audio_bytes),
                        media_type=f"audio/{request.output_format}",
                    )
                else:
                    return StreamingResponse(
                        io.BytesIO(audio_bytes),
                        media_type=f"audio/{request.output_format}",
                    )

            except Exception as e:
                self._update_metrics(success=False)
                logger.error(f"Synthesis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/morph")
        async def morph_voices(request: MorphRequest):
            """Synthesize with morphed voices."""
            try:
                if not self._check_rate_limit():
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

                start_time = time.time()
                audio = self.tts.morph_voices(
                    text=request.text,
                    speaker_ids=request.speaker_ids,
                    speaker_weights=request.speaker_weights,
                    emotion_ids=request.emotion_ids,
                    emotion_weights=request.emotion_weights,
                    speed=request.speed,
                )
                latency = time.time() - start_time

                self._update_metrics(success=True, latency=latency)

                audio_bytes = self._convert_audio(audio, request.output_format)
                return StreamingResponse(
                    io.BytesIO(audio_bytes), media_type=f"audio/{request.output_format}"
                )

            except Exception as e:
                self._update_metrics(success=False)
                logger.error(f"Morph error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/clone")
        async def clone_voice(
            request: VoiceCloneRequest, reference_audio: UploadFile = File(...)
        ):
            """Clone voice from reference audio."""
            try:
                if not self._check_rate_limit():
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

                # Read reference audio
                audio_data = await reference_audio.read()
                reference_array = self._load_audio_from_bytes(audio_data)

                start_time = time.time()
                audio = self.tts.clone_voice(
                    text=request.text,
                    reference_audio=reference_array,
                    emotion=request.emotion,
                )
                latency = time.time() - start_time

                self._update_metrics(success=True, latency=latency)

                audio_bytes = self._convert_audio(audio, request.output_format)
                return StreamingResponse(
                    io.BytesIO(audio_bytes), media_type=f"audio/{request.output_format}"
                )

            except Exception as e:
                self._update_metrics(success=False)
                logger.error(f"Clone error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch")
        async def batch_synthesize(request: BatchTTSRequest):
            """Batch synthesis."""
            try:
                if not self._check_rate_limit():
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

                if len(request.requests) > self.config.max_batch_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Batch size exceeds maximum of {self.config.max_batch_size}",
                    )

                results = []
                if request.parallel:
                    # Parallel processing
                    tasks = [self._generate_audio(req) for req in request.requests]
                    audios = await asyncio.gather(*tasks)
                else:
                    # Sequential processing
                    audios = []
                    for req in request.requests:
                        audio = await self._generate_audio(req)
                        audios.append(audio)

                # Convert all to bytes
                for i, (audio, req) in enumerate(zip(audios, request.requests)):
                    audio_bytes = self._convert_audio(audio, req.output_format)
                    # Encode as base64 for JSON response
                    import base64

                    results.append(
                        {
                            "index": i,
                            "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                            "format": req.output_format,
                        }
                    )

                return {"results": results}

            except Exception as e:
                self._update_metrics(success=False)
                logger.error(f"Batch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time synthesis."""
            await websocket.accept()
            try:
                while True:
                    # Receive request
                    data = await websocket.receive_json()
                    request = TTSRequest(**data)

                    # Generate audio
                    audio = await self._generate_audio(request)
                    audio_bytes = self._convert_audio(audio, "wav")

                    # Send response
                    import base64

                    await websocket.send_json(
                        {
                            "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                            "format": "wav",
                        }
                    )

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.close()

        @self.app.get("/speakers")
        async def list_speakers():
            """List available speakers."""
            # This would be loaded from a config file in practice
            speakers = [
                {"id": 0, "name": "Default", "gender": "female", "language": "ja"},
                {"id": 1, "name": "Male 1", "gender": "male", "language": "ja"},
                # ... more speakers
            ]
            return {"speakers": speakers}

        @self.app.get("/emotions")
        async def list_emotions():
            """List available emotions."""
            emotions = [
                "neutral",
                "happy",
                "sad",
                "angry",
                "surprised",
                "fear",
                "disgust",
            ]
            return {"emotions": emotions}

        @self.app.get("/styles")
        async def list_styles():
            """List available speaking styles."""
            styles = [
                "normal",
                "energetic",
                "calm",
                "dramatic",
                "whisper",
                "shout",
                "formal",
                "casual",
            ]
            return {"styles": styles}

    async def _generate_audio(self, request: TTSRequest) -> np.ndarray:
        """Generate audio from request."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            self.tts.synthesize,
            request.text,
            request.speaker_id,
            request.emotion,
            request.style,
            request.speed,
            request.pitch_shift,
            request.energy,
        )
        return audio

    def _convert_audio(self, audio: np.ndarray, format: str) -> bytes:
        """Convert audio to requested format."""
        buffer = io.BytesIO()

        if format == "wav":
            sf.write(buffer, audio, 48000, format="WAV")
        elif format == "mp3":
            # Would use pydub or ffmpeg for MP3
            # For now, just use WAV
            sf.write(buffer, audio, 48000, format="WAV")
        elif format == "ogg":
            # Would use pydub or ffmpeg for OGG
            sf.write(buffer, audio, 48000, format="WAV")
        else:
            raise ValueError(f"Unsupported format: {format}")

        buffer.seek(0)
        return buffer.read()

    def _load_audio_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Load audio from bytes."""
        buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buffer)
        return audio

    async def _stream_audio(self, audio_bytes: bytes):
        """Stream audio in chunks."""
        chunk_size = 1024 * 16  # 16KB chunks
        buffer = io.BytesIO(audio_bytes)

        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def _get_cache_key(self, request: TTSRequest) -> str:
        """Generate cache key from request."""
        key_data = f"{request.text}|{request.speaker_id}|{request.emotion}|{request.style}|{request.speed}|{request.pitch_shift}|{request.energy}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[bytes]:
        """Get cached result."""
        try:
            if self.cache.exists(key):
                return self.cache.get(key).encode("latin-1")
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None

    def _cache_result(self, key: str, data: bytes):
        """Cache result."""
        try:
            self.cache.setex(key, self.config.cache_ttl, data.decode("latin-1"))
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        now = time.time()
        # Clean old entries
        while self.request_history and self.request_history[0] < now - 60:
            self.request_history.popleft()

        if len(self.request_history) >= self.config.rate_limit:
            return False

        self.request_history.append(now)
        return True

    def _update_metrics(self, success: bool, latency: float = 0):
        """Update server metrics."""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
            # Update average latency
            n = self.metrics["successful_requests"]
            avg = self.metrics["average_latency"]
            self.metrics["average_latency"] = (avg * (n - 1) + latency) / n
        else:
            self.metrics["failed_requests"] += 1

    def run(self):
        """Run the server."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.num_workers,
        )


def create_server(config: Optional[ServerConfig] = None) -> InferenceServer:
    """Create inference server instance."""
    if config is None:
        config = ServerConfig()
    return InferenceServer(config)


if __name__ == "__main__":
    # Example usage
    server = create_server()
    server.run()
