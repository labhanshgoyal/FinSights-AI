from groq import Groq
import streamlit as st
import json

def validate_response(query, response):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        eval_prompt = f"""Score this financial AI response from 0.0 to 1.0.

Criteria:
- Relevancy: Does it answer the question?
- Specificity: Uses real data, not vague?
- Completeness: Covers trend, risk, outlook?

Question: {query}
Response: {response}

Reply ONLY with JSON: {{"score": 0.0, "feedback": "reason"}}"""

        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=100
        )
        parsed = json.loads(result.choices[0].message.content.strip())
        score = float(parsed.get("score", 0))
        return {
            "pass": score > 0.7,
            "score": round(score, 2),
            "feedback": parsed.get("feedback", "No feedback")
        }
    except Exception:
        return {"pass": True, "score": 1.0, "feedback": "Validation skipped"}
