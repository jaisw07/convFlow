import numpy as np
from typing import Tuple

from turn_taking.inference import predict_endpoint

class SmartTurnV3:
    """
    Thin wrapper around Smart Turn v3 inference.

    Responsibility:
    - Decide whether a user turn is complete
    - Return decision + confidence
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Probability thre shold for end-of-turn decision
        """
        self.threshold = threshold

    def is_end_of_turn(self, audio_8s: np.ndarray) -> Tuple[bool, float]:
        """
        Determine whether the user's turn has ended.

        Args:
            audio_8s: np.ndarray
                - 16 kHz
                - mono
                - float32
                - EXACTLY 8 seconds (128000 samples)
                - padded at the beginning

        Returns:
            (is_complete, probability)
        """

        if audio_8s.dtype != np.float32:
            raise ValueError("SmartTurnV3 expects float32 audio")

        if audio_8s.ndim != 1:
            raise ValueError("SmartTurnV3 expects 1D mono audio")

        if audio_8s.shape[0] != 16000 * 8:
            raise ValueError(
                f"SmartTurnV3 expects exactly 8 seconds of audio "
                f"(128000 samples), got {audio_8s.shape[0]}"
            )

        result = predict_endpoint(audio_8s)

        probability = float(result["probability"])
        is_complete = probability >= self.threshold

        return is_complete, probability
    
if __name__ == "__main__":
    import time

    from audio.mic_input import MicInput
    from audio.vad import SileroVAD
    from audio.buffer import TurnBuffer

    print("🎤 Speak into the microphone.")
    print("Smart Turn will decide when your turn is complete.\n")
    print("Press Ctrl+C to stop.\n")

    mic = MicInput()                 # 512-sample frames
    vad = SileroVAD()
    buffer = TurnBuffer()
    smart_turn = SmartTurnV3()

    def on_audio_frame(frame):
        is_speaking = vad.process_frame(frame)

        if is_speaking:
            buffer.add_speech_frame(frame)
        else:
            buffer.add_silence_frame(frame)

        if buffer.should_check_turn():
            audio_8s = buffer.get_audio_for_smart_turn()
            complete, prob = smart_turn.is_end_of_turn(audio_8s)

            print("🧠 Smart Turn check:")
            print(f"   Complete: {complete}")
            print(f"   Probability: {prob:.4f}\n")

            if complete:
                print("✅ Turn accepted. Resetting buffer.\n")
                buffer.reset()
            else:
                print("⏳ Not complete yet. Continuing to listen...\n")

    mic.start(on_audio_frame)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()