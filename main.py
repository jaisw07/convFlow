import asyncio
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import rtc
from livekit.rtc import Room, AudioStream
from livekit.api import AccessToken, VideoGrants
from audio.buffer import TurnBuffer
from audio.vad import SileroVAD
from stt.whisper_stt import WhisperSTT
from stt.progressive_stt import ProgressiveSTTController

LIVEKIT_URL = "ws://localhost:7880"
API_KEY = "devkey"
API_SECRET = "secret"

app = FastAPI()

# -------------------- CORS --------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Token --------------------

def create_token(identity: str, room_name: str):
    token = AccessToken(API_KEY, API_SECRET)
    token.with_identity(identity)
    token.with_grants(VideoGrants(room_join=True, room=room_name))
    return token.to_jwt()

@app.get("/token")
def token():
    return {"token": create_token("browser-user", "test-room")}

# -------------------- Progressive STT --------------------

whisper_stt = WhisperSTT()
progressive_stt = ProgressiveSTTController(whisper_stt)

# -------------------- Buffer --------------------

buffer = TurnBuffer(
    sample_rate=16000,
    max_turn_seconds=8.0,
    min_speech_seconds=1.5,
    silence_trigger_ms=1000,
    frame_duration_ms=10,
)

# ------------------- VAD ---------------------

vad = SileroVAD()

# vad expects 32ms frames but buffer collects 10ms frames -> rolling buffer added which collects till 32ms frames
vad_buffer = np.zeros(0, dtype=np.float32)
VAD_WINDOW_SAMPLES = int(16000 * 0.032)  # 32ms = 512 samples

# -------------------- Downsample Audio --------------------

def downsample_48k_to_16k(pcm_int16: np.ndarray) -> np.ndarray:
    # Convert to float32 in range [-1, 1]
    audio = pcm_int16.astype(np.float32) / 32768.0
    
    # Downsample by factor of 3 (48k → 16k)
    return audio[::3] 

# -------------------- LiveKit Agent --------------------

room = Room()

@app.on_event("startup")
async def startup():
    print("Starting LiveKit agent...")

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        print(f"Subscribed to {participant.identity}")

        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(handle_audio(track))

    await room.connect(
        LIVEKIT_URL,
        create_token("agent", "test-room")
    )

    print("Agent connected to room.")

async def handle_audio(track: rtc.RemoteAudioTrack):
    global vad_buffer

    stream = AudioStream(track)

    async for event in stream:
        frame = event.frame

        pcm_int16 = np.frombuffer(frame.data, dtype=np.int16)
        pcm_16k = downsample_48k_to_16k(pcm_int16)

        # ---- Accumulate until 32ms ----
        vad_buffer = np.concatenate([vad_buffer, pcm_16k])

        if len(vad_buffer) >= VAD_WINDOW_SAMPLES:
            vad_chunk = vad_buffer[:VAD_WINDOW_SAMPLES]
            vad_buffer = vad_buffer[VAD_WINDOW_SAMPLES:]

            is_speaking = vad.process_frame(vad_chunk)

            if is_speaking:
                buffer.add_speech_frame(vad_chunk)
            else:
                buffer.add_silence_frame(vad_chunk)

            asyncio.create_task(progressive_stt.maybe_process(buffer))

            if buffer.should_check_turn():
                print("🟢 Turn detected. Finalizing transcription...")
                transcript = await progressive_stt.finalize(buffer)

                print("\n📝 Final Transcript:")
                print(transcript)
                print("--------------------------------------------------\n")

                buffer.reset()