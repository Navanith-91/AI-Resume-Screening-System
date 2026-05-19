import streamlit as st

from utils.parser import extract_text_from_pdf
from utils.vector_store import create_vector_store, search_resume
from utils.llm import generate_response

st.title("AI Resume Screening System")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("Resume Uploaded Successfully!")

    extracted_text = extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted Resume Text")

    st.text_area(
        "Resume Content",
        extracted_text,
        height=250
    )

    # Create vector DB
    index, chunks = create_vector_store(extracted_text)

    st.success("FAISS Vector Database Created!")

    st.write(f"Total Chunks Created: {len(chunks)}")

    st.subheader("Ask Questions About Resume")

    user_query = st.text_input(
        "Enter your question"
    )

    if user_query:

        results = search_resume(
            user_query,
            index,
            chunks
        )

        context = " ".join(results)

        answer = generate_response(
            context,
            user_query
        )

        st.subheader("AI Response")

        st.write(answer)