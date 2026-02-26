import json
from llm.llm import GeminiLLM

class ProfileUpdater:
    def __init__(self):
        self.llm = GeminiLLM(
            temperature=0.0,
            system_prompt="Extract structured candidate profile updates. Respond ONLY in JSON."
        )

    def extract_updates(self, transcript: str, current_profile: dict):
        prompt = f"""
        Given the candidate response below, extract any new structured information.

        Candidate Response:
        {transcript}

        Current Profile:
        {current_profile}

        Return ONLY JSON:
        {{
            "name": string or null,
            "skills": list of strings,
            "projects": list of strings,
            "experience": list of strings
        }}

        Only include newly mentioned information.
        Do not hallucinate.
        """

        response = self.llm.generate(user_text=prompt)
        response = response.strip()

        if response.startswith("```"):
            response = response.strip("`")
            response = response.replace("json", "").strip()

        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1:
            response = response[start:end+1]

        try:
            return json.loads(response)
        except:
            return {}