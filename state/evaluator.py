import json
from llm.llm import GeminiLLM


class AnswerEvaluator:
    def __init__(self):
        self.llm = GeminiLLM(
            temperature=0.0,
            system_prompt=(
                "You are a strict technical interview evaluator."
                "You evaluate correctness, clarity, and depth."
                "Respond ONLY in valid JSON."
            )
        )

    def evaluate(self, question: str, answer: str) -> dict:
        prompt = f"""
            You are a strict but fair technical interview evaluator for fresher candidates.

            Question:
            {question}

            Candidate Answer:
            {answer}

            Return ONLY valid JSON. No explanation. No markdown.

            Return strictly:
            {{
                    "technical_depth": integer 0-10,
                    "correctness": integer 0-10,
                    "clarity": integer 0-10,
                    "confidence": integer 0-10,
                    "red_flags": list of short strings,
                    "followup_opportunities": list of short technical follow-up prompts
            }}
            """

        response = self.llm.generate(user_text=prompt)
        # 🔥 Strip possible markdown fences
        response = response.strip()
        
        if response.startswith("```"):
            response = response.strip("`")
            response = response.replace("json", "").strip()
        
        # 🔥 Extract first JSON block manually
        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1:
            json_str = response[start:end+1]
            try:
                return json.loads(json_str)
            except Exception:
                pass

        # fallback
        return {
            "technical_depth": 5,
            "correctness": 5,
            "clarity": 5,
            "confidence": 5,
            "red_flags": [],
            "followup_opportunities": []
        }