from audio.tts.piper_tts import PiperTTS
from audio.tts.kokoro_tts import KokoroTTS
from audio.tts.f5_tts import F5TTS

def create_tts(engine: str, **kwargs):
    """
    Create a TTS engine by name.

    Example:
        tts = create_tts(
            engine="piper",
            piper_bin="...",
            voice_model="..."
        )
    """

    engine = engine.lower()

    if engine == "piper":
        return PiperTTS(**kwargs)
    
    elif engine == "kokoro":
        return KokoroTTS(**kwargs)
    
    elif engine == "f5":
        return F5TTS(**kwargs)

    raise ValueError(f"Unknown TTS engine: {engine}")