# stt/whisper_stt.py

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import asyncio

SAMPLE_RATE = 16000

_gpu_lock = asyncio.Lock()

class WhisperSTT:
    def __init__(self, model_id: str = "Oriserve/Whisper-Hindi2Hinglish-Apex"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[WhisperSTT] Loading model on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.model.eval()

        print("[WhisperSTT] Model loaded.")

    def _transcribe_blocking(self, audio: np.ndarray) -> str:
        if audio.dtype != np.float32:
            raise ValueError("WhisperSTT expects float32 audio")

        if audio.ndim != 1:
            raise ValueError("WhisperSTT expects mono (1D) audio")

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        generated_ids = self.model.generate(
            input_features,
            task="transcribe",
            language="en",
        )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text.strip()

    async def transcribe_async(self, audio: np.ndarray) -> str:
        async with _gpu_lock:
            return await asyncio.to_thread(
                self._transcribe_blocking,
                audio
            )