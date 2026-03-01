"""
Orchestrator: Mic → VAD → Smart Turn → Whisper → Gemini → Piper TTS

This is the main entrypoint that ties the full pipeline together.
Barge-in is intentionally NOT implemented yet.
"""

import time
import queue
import threading
import numpy as np
import random

from audio.mic_input import MicInput
from audio.vad import SileroVAD
from audio.buffer import TurnBuffer
from turn_taking.smart_turn import SmartTurnV3
from stt.whisper_stt import WhisperSTT
from audio.tts.factory import create_tts

from state.state import InterviewState
from state.state import InterviewPhase
from stt.progressive_stt import ProgressiveSTTController

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
class TranscriptManager:
    def __init__(self):
        self.turns = []

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})

    def last_n(self, n=4):
        return self.turns[-n:]

    def full_text(self):
        return "\n".join([f"{t['role']}: {t['content']}" for t in self.turns])

RMS_SILENCE_THRESHOLD = 0.005

def summarize_transcript(llm, transcript_text):
    prompt = f"""
    Summarize the following interview in under 200 words.
    Focus on:
    - Candidate strengths
    - Technical signals
    - Gaps
    - Behavioral signals

    Transcript:
    {transcript_text}
    """
    return llm.generate(user_text=prompt)
# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    # --------------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------------

    mic = MicInput()
    vad = SileroVAD()
    buffer = TurnBuffer()
    smart_turn = SmartTurnV3()
    transcript_manager = TranscriptManager()
    progressive_stt = ProgressiveSTTController(stt)
    profile_updater = ProfileUpdater()

    stt = WhisperSTT()
    llm = GeminiLLM()
    evaluator = AnswerEvaluator()
    state = InterviewState(topic="SDE Intern Interview")

    # tts = create_tts(
    #     engine="piper",
    #     piper_bin=r"C:\Users\SHREY\Desktop\SpeechToText\piper_windows_amd64\piper\piper.exe",
    #     voice_model="audio/voices/en_GB-alan-low.onnx",
    # )

    tts = create_tts(
        engine="kokoro",
        lang_code="a",
        voice="af_heart",
        speed=1.0,
    )

    # tts = create_tts(
    #     engine="f5",
    #     ref_audio_path=r"C:\Users\SHREY\Documents\Sound Recordings\tanmay_sample.wav",
    #     ref_text="I'm just going to tell you my life's journey in very simple words, and which may not leave you inspired, but will help you survive this life.",
    #     device="cuda",
    #     speed = 1.0,
    #     nfe_step=24,   # lower = faster, slightly lower quality
    # )


    tts_busy = threading.Event()

    def on_tts_done():
        print("TTS Finished, Resuming Mic\n")
        tts_busy.clear()

    tts.on_done = on_tts_done

    # Queue of completed turns (audio)
    turn_queue = queue.Queue()

    # Conversation memory
    
    # check valid transcript or not
    def is_valid_transcript(text: str) -> bool:
        if not text:
            return False

        stripped = text.strip()

        if not stripped:
            return False

        lowered = stripped.lower()

        # Reject common junk outputs
        if lowered in {"nan", "none", "null"}:
            return False

        # Reject very short noise
        if len(stripped) < 2:
            return False

        return True

    def decide_mode(state: InterviewState) -> str:
        evaluation = state.last_evaluation

        if not evaluation:
            return "normal"

        correctness = evaluation.get("correctness", 5)
        depth = evaluation.get("technical_depth", 5)

        if correctness < 4:
            return "clarify_mistake"
        
        if state.followup_stack:
            return "use_followup"

        if depth < 5:
            return "probe_deeper"

        return "continue_topic"


    # --------------------------------------------------------------
    # Worker: STT → LLM → TTS
    # --------------------------------------------------------------

    async def turn_worker():
        while True:
            audio = turn_queue.get()
            if audio is None:
                break

            try:
                # ------------------ STT ------------------
                
                transcript = await progressive_stt.finalize(audio)
                if not is_valid_transcript(transcript):
                    print("⚠️ Invalid transcript detected.\n")
                    clarifications = [
                        "I'm sorry, I didn't quite catch that. Could you please repeat your response?",
                        "Apologies, there seems to have been some audio disruption. Would you mind repeating that?",
                        "I may have missed your last statement. Could you clarify your answer?",
                    ]
                    choice = random.choice(clarifications)
                    print(f"🤖 Gemini (clarification):\n{choice}\n")
                    tts_busy.set()
                    tts.speak(choice)
                    continue

                if state.phase == InterviewPhase.INTRO:
                    updates = profile_updater.extract_updates(
                    transcript,
                        {
                            "name": state.profile.name,
                            "skills": state.profile.skills,
                            "projects": state.profile.projects,
                            "experience": state.profile.experience
                        }
                    )
                    
                    if updates.get("name"):
                        state.profile.name = updates["name"]
                    
                    state.profile.skills.extend(updates.get("skills", []))
                    state.profile.projects.extend(updates.get("projects", []))
                    state.profile.experience.extend(updates.get("experience", []))

                    state.profile.update_missing_fields()

                
                transcript_manager.add_turn("user", transcript)

                print(f"📝 You said:\n{transcript}\n")
                # ------------------ EVALUATION ------------------

                if state.current_question is None:
                    # first turn: no evaluation
                    eval_result = {
                        "technical_depth": 5,
                        "correctness": 5,
                        "clarity": 5,
                        "confidence": 5,
                        "red_flags": [],
                        "followup_opportunities": []
                    }
                else:
                    eval_result = evaluator.evaluate(
                        question=state.current_question,
                        answer=transcript,
                    )

                print(f"📊 Evaluation: {eval_result}\n")
                behavior = eval_result.get("behavior", "normal")
                relevance = eval_result.get("relevance", 5)
                
                # 1️⃣ Record turn FIRST
                state.record_turn(
                    question=state.current_question or "Opening Question",
                    answer=transcript,
                    evaluation=eval_result,
                )

                # 2️⃣ Adjust difficulty
                state.adjust_difficulty()

                # 3️⃣ Now decide mode (uses updated evaluation)
                mode = decide_mode(state)

                followup_instruction = ""
                if mode == "use_followup" and state.followup_stack:
                    followup = state.followup_stack.pop(-1)
                    followup_instruction = f"Ask this follow-up question: {followup}. Set it up nicely for the interviewee by reiterating information to give context around it. It should not seem abrupt."
                
                if behavior == "abusive":
                    response = "Let's maintain professionalism. Please answer the question."
                    tts_busy.set()
                    tts.speak(response)
                    continue
                
                if behavior == "restart_attempt":
                    response = "We've already covered that earlier. Let's continue from where we left off."
                    tts_busy.set()
                    tts.speak(response)
                    continue

                if behavior == "irrelevant":
                    response = "Let's stay focused on the question. Could you address it directly?"
                    tts_busy.set()
                    tts.speak(response)
                    continue
                
                state.update_phase()

                # 4️⃣ Now build prompt
                adaptive_prompt = f"""
                    You are conducting a {state.topic}.
                    Current Phase: {state.phase.value}.

                    Phase rules:
                    INTRO → Gather name, skills, projects, experience.
                    TECHNICAL → Deep technical questions.
                    BEHAVIORAL → Situational + teamwork.
                    CLOSING → Wrap up.
                    You are a no-nonsense interviewer who is concerned about finding the highest quality candidate for the role. Do not coddle or appreciate unnecessarily.

                    Current difficulty level (1 is easiest, linear scale): {state.difficulty}
                    Branching mode: {mode}

                    Behavior classification: {behavior}
                    Relevance score: {relevance}
                    Current Phase: {state.phase.value}
                    Interview Summary:
                    {state.summary}

                    Rules:
                    - Do NOT repeat previous questions.
                    - Stay natural and conversational.

                    Behaviour Managing(prioritize over branching):
                    - If behavior is irrelevant → politely redirect.
                    - If restart_attempt → remind candidate we covered that.
                    - If erratic → stabilize conversation professionally.
                    - If abusive → warn once.

                    Branching behavior:
                    - clarify_mistake → Ask the candidate to correct or reconsider their answer.
                    - probe_deeper → Ask a deeper technical follow-up.
                    - increase_difficulty → Ask a harder, more advanced question.
                    - continue_topic → Continue the same topic naturally.
                    - use_followup → Use the provided follow-up question.

                    {followup_instruction}

                    Ask the next question.
                """

                short_memory = transcript_manager.last_n(4)

                response = llm.generate(
                    user_text=adaptive_prompt,
                    context=short_memory
                )

                if not response:
                    print("⚠️ Empty LLM response.\n")
                    continue

                print(f"🤖 Gemini:\n{response}\n")

                transcript_manager.add_turn("assistant", response)
                state.current_question = response


                # ------------------ TTS ------------------
                print("Pausing mic. TTS speaking.\n")
                tts_busy.set()
                tts.speak(response)

                # ------------------ Memory ------------------
                if state.turn_count % 3 == 0:
                    new_summary = summarize_transcript(llm, transcript_manager.full_text())
                    state.summary = new_summary

            finally:
                turn_queue.task_done()

    worker = threading.Thread(target=turn_worker, daemon=True)
    worker.start()

    # --------------------------------------------------------------
    # Audio callback
    # --------------------------------------------------------------
    from collections import deque

    smart_turn_probs = deque(maxlen=5)
    SMART_TURN_THRESHOLD = 0.75
    REQUIRED_HISTORY = 5

    pending_turn = {
        "active": False,
        "timestamp": 0.0,
        "audio": None
    }

    CONFIRM_WINDOW = 0.5 # 500ms

    FAILSAFE_SILENCE_SECONDS = 2
    FRAME_DURATION_SEC = 0.032
    FAILSAFE_SILENCE_FRAMES = int(FAILSAFE_SILENCE_SECONDS / FRAME_DURATION_SEC)


    async def on_audio_frame(frame: np.ndarray):
        # ignore mic while tts speaking
        if tts_busy.is_set():
            return
        
        is_speaking = vad.process_frame(frame)

        if is_speaking:
            buffer.add_speech_frame(frame)
        else:
            buffer.add_silence_frame(frame)

        await progressive_stt.maybe_process(buffer.get_full_turn_audio())

        # -------------------------------------------------
        # Hard silence failsafe (2s silence → force commit)
        # -------------------------------------------------

        if (
            buffer.speech_samples >= buffer.min_speech_samples
            and buffer._silent_frames >= FAILSAFE_SILENCE_FRAMES
            and not pending_turn["active"]  
        ):
            print("⏱ Failsafe triggered (2s silence). Forcing commit.")

            raw_audio = buffer.get_full_turn_audio()

            def rms(audio: np.ndarray) -> float:
                return np.sqrt(np.mean(audio ** 2))

            if rms(raw_audio) >= RMS_SILENCE_THRESHOLD:
                print(f"Turn duration: {len(raw_audio)/16000:.2f} sec")
                turn_queue.put(raw_audio)

            buffer.reset()
            smart_turn_probs.clear()
            pending_turn["active"] = False
            return


        if buffer.should_check_turn():
            audio_8s = buffer.get_audio_for_smart_turn()
            _, prob = smart_turn.is_end_of_turn(audio_8s)

            smart_turn_probs.append(prob)
            avg_prob = sum(smart_turn_probs) / len(smart_turn_probs)

            print("🧠 Smart Turn check:")
            print(f"   Instant Prob: {prob:.4f}")
            print(f"   Avg Prob: {avg_prob:.4f}")


            if (
                len(smart_turn_probs) == REQUIRED_HISTORY
                and avg_prob >= SMART_TURN_THRESHOLD
                and not pending_turn["active"]
            ):
                print("🟡 SmartTurn suggests turn end. Awaiting confirmation...")
                pending_turn["active"] = True
                pending_turn["timestamp"] = time.time()
                pending_turn["audio"] = buffer.get_full_turn_audio()
                return
            
            # ----------------------------------------
            # Confirm pending turn if silence continues
            # ----------------------------------------

            if pending_turn["active"]:
                elapsed = time.time() - pending_turn["timestamp"]

                if elapsed >= CONFIRM_WINDOW:
                    # If still silent → commit
                    if (
                        buffer._silent_frames >= buffer.silence_trigger_frames
                        and avg_prob >= SMART_TURN_THRESHOLD
                        ):
                        raw_audio = pending_turn["audio"]

                        def rms(audio: np.ndarray) -> float:
                            return np.sqrt(np.mean(audio ** 2))

                        if rms(raw_audio) < RMS_SILENCE_THRESHOLD:
                            print("⚠️ Skipping turn (audio too quiet)\n")
                            buffer.reset()
                        else:
                            print("✅ Turn confirmed. Processing...\n")
                            print(f"Turn duration: {len(raw_audio)/16000:.2f} sec")
                            turn_queue.put(raw_audio) 
                            buffer.reset()

                    else:
                        print("⚠️ Speech resumed. Cancelling turn.\n")
                        smart_turn_probs.clear()

                    pending_turn["active"] = False
                    smart_turn_probs.clear()



    # --------------------------------------------------------------
    # Start mic
    # --------------------------------------------------------------

    mic.start(on_audio_frame)

    
    # --- Add your welcome TTS here ---
    welcome_message = f"Welcome to your {state.topic}. Let's start off with a brief introduction about yourself."
    tts_busy.set()
    tts.speak(welcome_message)
    state.current_question = welcome_message

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        turn_queue.put(None)
        turn_queue.join()