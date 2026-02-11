import threading
import sounddevice as sd
import numpy as np

from kokoro import KPipeline
from audio.tts.base import BaseTTS


class KokoroTTS(BaseTTS):
    """
    Kokoro-82M TTS wrapper.

    Responsibilities:
    - Text → speech audio (chunked generator)
    - Play audio sequentially
    - Support interruption
    """

    SAMPLE_RATE = 24000  # Kokoro default

    def __init__(
        self,
        lang_code: str = "a",   # 'a' = American English
        voice: str = "af_heart",
        speed: float = 1.0,
    ):
        super().__init__()

        self.voice = voice
        self.speed = speed

        print("[KokoroTTS] Loading pipeline...")
        self.pipeline = KPipeline(lang_code=lang_code)
        print("[KokoroTTS] Initialized.")

        self._play_thread = None
        self._stop_event = threading.Event()

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def speak(self, text: str):
        if not text.strip():
            return

        self.stop()
        self._stop_event.clear()

        self._play_thread = threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True,
        )
        self._play_thread.start()

    def stop(self):
        if self._play_thread and self._play_thread.is_alive():
            self._stop_event.set()
            sd.stop()

    # --------------------------------------------------------------
    # Internal
    # --------------------------------------------------------------

    def _speak_worker(self, text: str):
        try:
            generator = self.pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=r"\n+",
            )

            for i, (gs, ps, audio) in enumerate(generator):

                if self._stop_event.is_set():
                    break

                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)

                sd.play(audio, self.SAMPLE_RATE)
                sd.wait()

        finally:
            if self.on_done:
                self.on_done()