import sounddevice as sd
import numpy as np
from typing import Callable, Optional


class MicInput:
    """
    Real-time microphone audio capture using sounddevice.

    - Sample rate: 16 kHz
    - Channels: mono
    - Output format: np.ndarray (float32, shape: [n_samples])
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 32,
        device: Optional[int] = None,
    ):
        """
        Args:
            sample_rate: Target sampling rate (must be 16000 for Silero + Smart Turn)
            frame_duration_ms: Frame size in milliseconds (20–30ms recommended)
            device: Optional sounddevice input device index
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.device = device

        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.stream: Optional[sd.InputStream] = None

        self._callback: Optional[Callable[[np.ndarray], None]] = None

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Start microphone capture.

        Args:
            callback: Function called for each audio frame.
                      Receives np.ndarray of shape (frame_size,)
        """
        if self.stream is not None:
            raise RuntimeError("MicInput stream already running")

        self._callback = callback

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,                # mono
            dtype="float32",           # Silero expects float32
            blocksize=self.frame_size,
            device=self.device,
            callback=self._audio_callback,
        )

        self.stream.start()

    def stop(self) -> None:
        """Stop microphone capture."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, indata, frames, time, status):
        if status:
            # Non-fatal: underruns/overruns can happen
            print(f"[MicInput] Audio status: {status}")

        if self._callback is None:
            return

        # indata shape: (frames, channels)
        # Convert to 1D mono array
        frame = np.squeeze(indata, axis=1)

        # Defensive copy (sounddevice reuses buffers)
        self._callback(frame.copy())

""" if __name__ == "__main__":
    mic = MicInput()

    def debug_callback(frame):
        print(frame.shape, frame.dtype)

    mic.start(debug_callback)
    input("Recording... press Enter to stop\n")
    mic.stop() """
    