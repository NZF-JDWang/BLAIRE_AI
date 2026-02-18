from pydantic import BaseModel, Field


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class TtsResponse(BaseModel):
    audio_base64: str
    sample_rate: int


class SttResponse(BaseModel):
    text: str
