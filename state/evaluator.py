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
        prompt = f"""Evaluate the candidate answer. 
                        Question: {question} 
                        Answer: {answer}
                        Return ONLY valid JSON.
                        Do NOT include any text before or after the JSON.
                        Do NOT explain anything.
                        Return strictly:
                            {{
                                "score": float between 0 and 1,
                                "depth": "low" | "medium" | "high",
                                "feedback": "short explanation"
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
            "score": 0.5,
            "depth": "medium",
            "feedback": "Evaluation parsing failed."
        }