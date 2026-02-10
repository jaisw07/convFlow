from audio.tts.piper_tts import PiperTTS


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

    raise ValueError(f"Unknown TTS engine: {engine}")