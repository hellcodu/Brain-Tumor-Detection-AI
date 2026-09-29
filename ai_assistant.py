import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = None

if api_key:
    client = OpenAI(
        api_key=api_key
    )


# =====================================================
# LOCAL FALLBACK - REPORT EXPLANATION
# =====================================================

def local_report_explanation(
    prediction,
    confidence,
    probabilities=None
):

    return f"""
AI Report Explanation

The AI model classified this MRI image as:

Prediction: {prediction}

Confidence: {confidence}%

The confidence score indicates how strongly the
AI model selected this category compared with the
other categories it was trained to recognize.

This is an AI-assisted image classification result.
It is not a medical diagnosis and should not be
considered definitive.

The MRI should be reviewed by a qualified healthcare
professional who can consider the patient's symptoms,
medical history, and the original medical images.

If you have concerns about this result, please
discuss it with a doctor or radiologist.
"""


# =====================================================
# AI REPORT EXPLANATION
# =====================================================

def explain_report(
    prediction,
    confidence,
    probabilities=None
):

    prompt = f"""
You are an educational AI assistant
for a brain MRI report system.

Prediction:
{prediction}

Confidence:
{confidence}%

Class probabilities:
{probabilities}

Explain the AI-generated result in
simple language.

Explain:
1. What the prediction means.
2. What the confidence score means.
3. What the user should understand.

Do not diagnose the patient.
Do not prescribe treatment.
Do not claim the AI result is definitive.
Recommend professional medical review.
"""

    # ---------------------------------------------
    # Try OpenAI
    # ---------------------------------------------

    if client is not None:

        try:

            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt
            )

            return response.output_text

        except Exception as e:

            print(
                "OpenAI API unavailable:",
                e
            )

    # ---------------------------------------------
    # Fallback
    # ---------------------------------------------

    print(
        "Using local report explanation."
    )

    return local_report_explanation(
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities
    )


# =====================================================
# AI CHAT
# =====================================================

def chat_with_ai(
    question,
    prediction=None,
    confidence=None
):

    return chat_about_report(
        question=question,
        prediction=prediction,
        confidence=confidence
    )


# =====================================================
# LOCAL CHAT FALLBACK
# =====================================================

def local_chat_response(
    question,
    prediction=None,
    confidence=None
):

    return f"""
I can explain the AI-generated MRI result in
general educational terms.

Current AI prediction:
{prediction}

Confidence:
{confidence}%

Your question was:
{question}

Please remember that this system provides
AI-assisted image classification only. It cannot
diagnose a medical condition or recommend treatment.

For interpretation of an MRI and decisions about
your health, please consult a qualified doctor or
radiologist.
"""


# =====================================================
# CHAT ABOUT REPORT
# =====================================================

def chat_about_report(
    question,
    prediction=None,
    confidence=None,
    ai_explanation=None
):

    prompt = f"""
You are an educational AI assistant
for a brain MRI report system.

AI prediction:
{prediction}

Confidence:
{confidence}%

AI explanation:
{ai_explanation}

User question:
{question}

Answer clearly and simply.

Important:
- Do not diagnose the patient.
- Do not prescribe treatment.
- Do not claim the AI prediction is definitive.
- Encourage consultation with a qualified
  healthcare professional.
"""

    # ---------------------------------------------
    # Try OpenAI
    # ---------------------------------------------

    if client is not None:

        try:

            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt
            )

            return response.output_text

        except Exception as e:

            print(
                "OpenAI API unavailable:",
                e
            )

    # ---------------------------------------------
    # Fallback
    # ---------------------------------------------

    print(
        "Using local AI chat fallback."
    )

    return local_chat_response(
        question=question,
        prediction=prediction,
        confidence=confidence
    )