import base64
from pathlib import Path
from shutil import which
import subprocess
import tempfile

from app.core.config import Settings


class VoiceServiceError(RuntimeError):
    pass


class VoiceService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def synthesize_tts(self, text: str) -> tuple[str, int]:
        piper = self._settings.piper_bin
        voice_model = self._settings.piper_voice_model.strip()
        if not voice_model:
            raise VoiceServiceError("PIPER_VOICE_MODEL is not configured")
        if which(piper) is None:
            raise VoiceServiceError("Piper binary is not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "speech.wav"
            cmd = [piper, "--model", voice_model, "--output_file", str(out_path)]
            try:
                proc = subprocess.run(  # noqa: S603
                    cmd,
                    input=text,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except Exception as exc:  # noqa: BLE001
                raise VoiceServiceError(f"TTS execution failed: {exc}") from exc
            if proc.returncode != 0 or not out_path.exists():
                raise VoiceServiceError("Piper synthesis failed")
            encoded = base64.b64encode(out_path.read_bytes()).decode("ascii")
        return encoded, 22050

    def transcribe_stt(self, audio_path: Path) -> str:
        whisper = self._settings.faster_whisper_bin
        if which(whisper) is None:
            raise VoiceServiceError("faster-whisper binary is not available")
        cmd = [whisper, str(audio_path), "--model", self._settings.faster_whisper_model]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)  # noqa: S603
        except Exception as exc:  # noqa: BLE001
            raise VoiceServiceError(f"STT execution failed: {exc}") from exc
        if proc.returncode != 0:
            raise VoiceServiceError("faster-whisper transcription failed")
        transcript = proc.stdout.strip()
        if not transcript:
            raise VoiceServiceError("No transcription output")
        return transcript
