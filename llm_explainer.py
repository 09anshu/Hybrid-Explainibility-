from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def get_diagnosis_verdicts(predictions):
    """Formats model predictions into human readable text."""
    diagnosis_verdicts = []
    for disease, prob in predictions.items():
        if prob > 0.5:
            risk = "High Risk (Positive)"
        elif prob > 0.2:
            risk = "Moderate / Elevated Risk"
        else:
            risk = "Low Risk (Negative)"
        diagnosis_verdicts.append(f"{disease}: {risk} ({prob:.1%} probability)")
    return "\n".join(diagnosis_verdicts)


def format_system_prompt(prob_text, regions_text):
    return f"""You are a helpful expert radiologist AI assistant.
Discuss this chest X-ray with the patient using a structured, reasoning-based approach.

**Factual Deductions from AI Model:**
{prob_text}

**Actual Visual Evidence (Grad-CAM Focus):**
{regions_text}

**Your Task:**
Provide your response in the following structured format:

1. **Findings Summary:** Briefly summarize the model's factual deductions and risk labels. Mention any extreme class imbalances if the percentages are low but the model flags it as "Elevated Risk".
2. **Visual Evidence Reasoning:** Explain *why* the model might have looked at the specific regions highlighted by Grad-CAM (e.g., "The model focused on the lower lung zones, which is typically where fluid accumulation from edema or consolidations from pneumonia are spotted.").
3. **Medical Impression:** Deliver a clear, empathetic, and professional explanation of what this means for the patient.

If the risk is low, reassure them. If elevated, explain the findings clearly without being alarmist.
"""


GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]


def get_gemini_chain(api_key, model_name=None):
    """Returns a LangChain pipeline using Google Gemini.

    If *model_name* is None, tries GEMINI_MODELS in order until one
    responds successfully.  Returns (chain, model_used).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    models_to_try = [model_name] if model_name else GEMINI_MODELS
    last_err = None

    for m in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=m,
                temperature=0.3,
                max_output_tokens=1024,
                google_api_key=api_key,
            )
            # Quick probe to verify the model responds
            llm.invoke("ping")
            return prompt | llm, m
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"All Gemini models exhausted. Last error: {last_err}"
    )


# ── Backward-compat alias ─────────────────────────────────────────────────
def get_huggingface_chain(api_key):
    """Deprecated — kept for backward compatibility. Use get_gemini_chain()."""
    return get_gemini_chain(api_key)


if __name__ == "__main__":
    print("LangChain Gemini Explainer Module ready.")
