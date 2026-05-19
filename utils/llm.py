from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

def generate_response(context, question):

    prompt = f"""
    You are an AI Resume Screening Assistant.

    Resume Information:
    {context}

    Question:
    {question}

    Give a professional answer.
    """

    result = generator(
        prompt,
        max_new_tokens=100
    )

    return result[0]['generated_text']