import numpy as np

from kokoro import KPipeline
from audio.tts.base import BaseTTS

class KokoroTTS(BaseTTS):

    SAMPLE_RATE = 24000

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        speed: float = 1.0,
    ):
        super().__init__()
        self.voice = voice
        self.speed = speed
        self.pipeline = KPipeline(lang_code=lang_code)

    def speak(self, text: str):
        pass

    def stop(self):
        pass

    def synthesize(self, text: str):
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r"\n+",
        )

        for _, _, audio in generator:
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            yield audio