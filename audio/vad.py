import torch
import numpy as np
from typing import Optional


class SileroVAD:
    """
    Lightweight real-time Voice Activity Detection using Silero VAD.

    This class answers one question:
    → "Is the user currently speaking?"
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        min_silence_ms: int = 500,
        frame_duration_ms: int = 32,
    ):
        """
        Args:
            sample_rate: Audio sample rate (must be 16000)
            speech_threshold: Probability threshold for speech detection
            min_silence_ms: How long silence must last to count as silence
            frame_duration_ms: Duration of each audio frame
        """

        if sample_rate != 16000:
            raise ValueError("Silero VAD requires 16kHz audio")

        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold

        # How many silent frames in a row mean "real silence"
        self.min_silence_frames = int(min_silence_ms / frame_duration_ms)

        self._silent_frames = 0
        self._is_speaking = False

        # Load Silero VAD model
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

        self.model.eval()

    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process a single audio frame.

        Args:
            frame: np.ndarray, shape (n_samples,), float32

        Returns:
            bool: True if user is speaking, False otherwise
        """

        # Convert numpy → torch
        audio_tensor = torch.from_numpy(frame)

        # Silero returns speech probability [0.0, 1.0]
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        is_speech = speech_prob >= self.speech_threshold

        if is_speech:
            self._silent_frames = 0
            self._is_speaking = True
        else:
            self._silent_frames += 1
            if self._silent_frames >= self.min_silence_frames:
                self._is_speaking = False

        return self._is_speaking

    def reset(self) -> None:
        """Reset internal state (call after a turn completes)."""
        self._silent_frames = 0
        self._is_speaking = False


""" if __name__ == "__main__":
    from mic_input import MicInput
    import time

    vad = SileroVAD()
    mic = MicInput()

    print("🎤 Speak into the mic. Silence will be detected automatically.")
    print("Press Ctrl+C to stop.\n")

    last_state = None

    def on_audio_frame(frame):
        global last_state

        is_speaking = vad.process_frame(frame)

        # Print only when state changes (to avoid spam)
        if is_speaking != last_state:
            if is_speaking:
                print("🟢 Speaking")
            else:
                print("🔴 Silence")
            last_state = is_speaking

    mic.start(on_audio_frame)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop() """