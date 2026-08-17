import streamlit as st
import requests

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Advanced RAG",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------
# Title
# ---------------------------------
st.title("🤖 Advanced RAG Assistant")
st.write("Ask questions from your document knowledge base.")

# ---------------------------------
# FastAPI URL
# ---------------------------------
API_URL = "http://127.0.0.1:8000/query"

# ---------------------------------
# Question Input
# ---------------------------------
question = st.text_input(
    "Enter your question:",
    placeholder="What is unsupervised learning?"
)

# ---------------------------------
# Ask Button
# ---------------------------------
if st.button("Ask", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Retrieving information and generating answer..."):

            try:
                # Send request to FastAPI
                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=120
                )

                # Raise error for 4xx / 5xx responses
                response.raise_for_status()

                # Convert response to JSON
                result = response.json()

                # ---------------------------------
                # Display Response
                # ---------------------------------
                st.subheader("Answer")

                # Try common response formats
                if isinstance(result, dict):

                    if "answer" in result:
                        st.write(result["answer"])

                    elif "response" in result:
                        st.write(result["response"])

                    else:
                        st.json(result)

                else:
                    st.write(result)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Could not connect to FastAPI. "
                    "Make sure FastAPI is running on port 8000."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ The request took too long to complete."
                )

            except requests.exceptions.HTTPError as e:
                st.error(
                    f"❌ FastAPI returned an HTTP error: {e}"
                )

                # Show FastAPI's error response
                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)

            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")