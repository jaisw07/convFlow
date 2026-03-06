# 🎙️ Real‑Time AI Interview Voice Agent (v2 Architecture)

This document explains the **updated architecture and logical flow** for
the real‑time AI technical interview system.

The design focuses on:

-   Low latency voice interaction
-   Parallel reasoning
-   Deterministic interview structure
-   Token‑efficient context management
-   Streaming interviewer responses
-   Scalable architecture for production

------------------------------------------------------------------------

# 🧠 Interview Flow Overview

The interview follows a deterministic sequence of phases.

``` mermaid
flowchart TD
    A[Intro] --> B[Resume / Projects]
    B --> C[Core CS]
    C --> D[Scenario Based]
    D --> E[Candidate Questions]
    E --> F[Closure]
```

### Phase Explanation

**Intro** - Candidate introduction - Background summary

**Resume / Projects** - Candidate explains past projects - Architecture
decisions - Technology choices

**Core CS** - Data structures - Algorithms - Computer science
fundamentals

**Scenario Based** - Debugging scenarios - System reasoning problems

**Candidate Questions** - Candidate can ask about role or company

**Closure** - Interview wrap‑up

------------------------------------------------------------------------

# 🎧 Voice Pipeline

The voice stack converts speech → reasoning → streamed speech.

``` mermaid
flowchart LR
    Mic --> LiveKit
    LiveKit --> VAD
    VAD --> TurnBuffer
    TurnBuffer --> ProgressiveSTT
    ProgressiveSTT --> InterviewEngine
    InterviewEngine --> StreamingLLM
    StreamingLLM --> SentenceChunker
    SentenceChunker --> TTS
    TTS --> LiveKitAudio
```

### Components

  Component          Role
  ------------------ ----------------------------------
  LiveKit            WebRTC audio transport
  Silero VAD         Voice activity detection
  TurnBuffer         Speech turn detection
  Progressive STT    Streaming transcription
  Interview Engine   Interview reasoning
  Gemini LLM         Decision and question generation
  Sentence Chunker   Splits tokens into TTS sentences
  Kokoro TTS         Speech synthesis

------------------------------------------------------------------------

# ⚙️ Turn Processing Architecture

Every **candidate answer triggers the reasoning pipeline**.

``` mermaid
flowchart TD
    A[User Answer]
    A --> B1[Node A<br>Evaluation + Behaviour Detection]
    A --> B2[Node B<br>Topic & Phase Decision]

    B1 --> C[Node C Streaming Response Generator]
    B2 --> C

    C --> D[TTS Streaming]
```

### Key Principle

**Node A and Node B run in parallel.**

This reduces latency and enables faster interviewer responses.

------------------------------------------------------------------------

# 🔎 Node A --- Evaluation & Behaviour Detection

### Inputs

-   last_question
-   last_answer
-   rolling_summary

### Responsibilities

Evaluate the candidate answer and detect abnormal behaviour.

Evaluation considers:

-   correctness
-   depth of explanation
-   clarity of language

### Output

``` json
{
 "score": 7,
 "unexpFlag": false,
 "unexpDesc": ""
}
```

If unexpected behaviour occurs:

``` json
{
 "score": 0,
 "unexpFlag": true,
 "unexpDesc": "candidate attempted to escape interview simulation"
}
```

Examples of unexpected behaviour:

-   abusive language
-   irrelevant answers
-   attempts to reset interview
-   refusal to answer

------------------------------------------------------------------------

# 🧭 Node B --- Topic & Phase Decision

Node B decides the **next question direction**.

### Inputs

-   last_question
-   last_answer
-   rolling_summary
-   asked_questions_phase
-   phase
-   question_index
-   followup_count

### Responsibilities

Determine:

-   follow‑up vs new topic
-   next topic to ask
-   phase transitions

### Output

``` json
{
 "nextTopic": "followup",
 "desc": "ask candidate to explain why learning rate affects convergence",
 "nextPhase": "core_cs"
}
```

Example new topic decision:

``` json
{
 "nextTopic": "hash tables",
 "desc": "ask candidate about collision resolution strategies",
 "nextPhase": "core_cs"
}
```

------------------------------------------------------------------------

# 🗣 Node C --- Streaming Response Generator

Node C produces the **spoken interviewer response**.

### Input Logic

If unexpected behaviour is detected:

    Node C receives:
    unexpFlag
    unexpDesc

Else Node C receives Node B instructions:

    nextTopic
    desc
    nextPhase

### Behaviour

If unexpected:

-   respond professionally
-   redirect candidate

Otherwise:

-   ask question based on Node B plan

------------------------------------------------------------------------

# 🔊 Streaming Response Pipeline

The interviewer response is **generated token‑by‑token**.

``` mermaid
flowchart LR
    LLMStream --> SentenceChunker
    SentenceChunker --> TTSQueue
    TTSQueue --> AudioPlayback
```

### Benefits

-   early speech playback
-   lower perceived latency
-   natural conversation flow

------------------------------------------------------------------------

# 🧾 Rolling Transcript Summary

Instead of storing full conversation history, the system maintains a
**rolling summary**.

### Summary Structure

    Q: ...
    A: ...
    Q: ...
    A: ...

### Trigger Conditions

Summary compression runs when:

-   **3 Q&A turns**, OR
-   **1000 token limit reached**

``` mermaid
flowchart TD
    A[Append New Q&A]
    A --> B{Trigger Condition}

    B -->|No| C[Continue Interview]
    B -->|Yes| D[Background Summarizer]

    D --> E[Compressed Rolling Summary]
```

### Important Property

The summarizer runs **asynchronously in the background**, so the
interview **never pauses**.

------------------------------------------------------------------------

# 🧩 Interview State

The system maintains a lightweight state object.

    rolling_summary
    asked_questions_phase

    last_question
    last_answer

    candidate_profile

    phase
    question_index
    followup_count

### Why this design

Benefits:

-   smaller prompts
-   predictable token usage
-   faster LLM calls

------------------------------------------------------------------------

# ⚡ Latency Strategy

The system minimizes response time through:

-   Parallel reasoning nodes
-   Rolling summarization
-   Token‑efficient prompts
-   Streaming generation
-   Sentence‑level TTS playback

Typical timeline:

    STT finalize
    ↓
    Node A + Node B (parallel)
    ↓
    Node C streaming generation
    ↓
    Sentence chunk → TTS
    ↓
    Audio playback

First audio latency target:

**\< 1 second**

------------------------------------------------------------------------

# 🧱 Token‑Oriented Object Notation (TOON)

TOON is used to reduce prompt size.

Example input to Node B:

    InterviewContext{
     phase:core_cs
     qIndex:2
     followups:1
     lastQ:"Explain gradient descent"
     lastA:"..."
     summary:"candidate discussed CNN training"
     askedQ:["gradient descent","activation functions"]
    }

Advantages:

-   token efficiency
-   deterministic structure
-   faster LLM parsing

------------------------------------------------------------------------

# 🚀 System Advantages

This architecture provides:

-   Real‑time conversational interviewing
-   Parallel reasoning for low latency
-   Token‑efficient context management
-   Robust behaviour detection
-   Natural streaming interviewer responses

The result is a **production‑grade AI technical interviewer capable of
realistic voice interviews.**
