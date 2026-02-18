from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.voice import SttResponse, TtsRequest, TtsResponse
from app.services.voice_service import VoiceService, VoiceServiceError

router = APIRouter(tags=["voice"])


@router.post("/voice/tts", response_model=TtsResponse)
async def tts(
    request: TtsRequest,
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> TtsResponse:
    service = VoiceService(get_settings())
    try:
        audio_base64, sample_rate = service.synthesize_tts(request.text)
    except VoiceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TtsResponse(audio_base64=audio_base64, sample_rate=sample_rate)


@router.post("/voice/stt", response_model=SttResponse)
async def stt(
    file: UploadFile = File(...),
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> SttResponse:
    service = VoiceService(get_settings())
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        text = service.transcribe_stt(temp_path)
    except VoiceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return SttResponse(text=text)
