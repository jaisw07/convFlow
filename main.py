import subprocess
import tempfile
import os
import sounddevice as sd
import soundfile as sf


# -----------------------------
# CONFIG — EDIT THESE
# -----------------------------

# REF_AUDIO = r"C:\Users\SHREY\Downloads\srk_clean_sample.wav"  # <-- path to your reference audio
# REF_TEXT = "I'm just going to tell you my life's journey in very simple words, and which may not leave you inspired, but will help you survive this life. And if you can do that, kids, if you can survive, happiness, creativity and success will follow on its own."
# REF_AUDIO = r"C:\Users\SHREY\Documents\Sound Recordings\tanmay_sample.wav"  # <-- path to your reference audio
# REF_TEXT = "I'm just going to tell you my life's journey in very simple words, and which may not leave you inspired, but will help you survive this life."
REF_AUDIO = r"C:\Users\SHREY\Desktop\Standard recording 8.wav"  # <-- path to your reference audio
REF_TEXT = "My name is Jane Doe and this is the clean recording of my words."
MODEL_NAME = "F5TTS_v1_Base"
DEVICE = "cuda"  # change to "cpu" if no GPU


# -----------------------------
# MAIN
# -----------------------------

def main():

    if not os.path.exists(REF_AUDIO):
        print("Reference audio not found:", REF_AUDIO)
        return

    with tempfile.TemporaryDirectory() as tmpdir:

        output_file = "output.wav"

        cmd = [
            "f5-tts_infer-cli",
            "--model", MODEL_NAME,
            "--ref_audio", REF_AUDIO,
            "--ref_text", REF_TEXT,
            "--gen_text", "Good morning, thanks for coming in today. Could you start by briefly introducing yourself and telling us about your background? We’d also like to understand what interested you in this role and how your previous experience prepares you for this position. Finally, could you describe a challenging project you worked on recently and how you handled any difficulties that came up during it?",
            "--output_dir", tmpdir,
            "--output_file", output_file,
            "--device", DEVICE,
        ]

        print("\nRunning F5 CLI...\n")
        print(" ".join(cmd))
        print()

        process = subprocess.Popen(cmd)
        process.wait()

        wav_path = os.path.join(tmpdir, output_file)

        if not os.path.exists(wav_path):
            print("\n❌ Output file not found.")
            return

        print("\n✅ Generation complete. Playing audio...\n")

        audio, sr = sf.read(wav_path, dtype="float32")
        sd.play(audio, sr)
        sd.wait()

        print("\nDone.")


if __name__ == "__main__":
    main()