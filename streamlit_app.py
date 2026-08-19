import streamlit as st
import requests
import csv
import io
import time
import os
# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# FASTAPI CONFIGURATION
# ==========================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/query"
)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "export_data" not in st.session_state:
    st.session_state.export_data = None


# ==========================================
# FUNCTION: CREATE CHAT EXPORT
# ==========================================

def create_chat_export(messages):

    output = io.StringIO()

    writer = csv.writer(output)

    # CSV header
    writer.writerow([
        "Question",
        "Retrieved Sources"
    ])

    current_question = None

    for message in messages:

        # Store user question
        if message["role"] == "user":

            current_question = message["content"]

        # Store sources associated with assistant response
        elif message["role"] == "assistant":

            sources = message.get("sources", [])

            source_text = ""

            if sources:

                source_list = []

                for source in sources:

                    if isinstance(source, dict):

                        # Try to extract useful information
                        source_info = []

                        if "source" in source:
                            source_info.append(
                                f"File: {source['source']}"
                            )

                        if "page" in source:
                            source_info.append(
                                f"Page: {source['page']}"
                            )

                        if "content" in source:
                            source_info.append(
                                source["content"]
                            )

                        if source_info:
                            source_list.append(
                                " | ".join(source_info)
                            )
                        else:
                            source_list.append(
                                str(source)
                            )

                    else:
                        source_list.append(str(source))

                source_text = "\n\n".join(source_list)

            writer.writerow([
                current_question or "",
                source_text
            ])

            current_question = None

    return output.getvalue()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("⚙️ RAG Assistant")

    st.write(
        "Advanced RAG system powered by "
        "Streamlit + FastAPI."
    )

    st.divider()

    # --------------------------------------
    # CLEAR CHAT
    # --------------------------------------

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        # Create export BEFORE clearing messages
        if st.session_state.messages:

            st.session_state.export_data = (
                create_chat_export(
                    st.session_state.messages
                )
            )

        # Clear chat
        st.session_state.messages = []

        st.rerun()

    st.divider()

    # --------------------------------------
    # DOWNLOAD EXPORT
    # --------------------------------------

    if st.session_state.export_data:

        st.subheader("📥 Chat Export")

        st.download_button(
            label="Download Chat CSV",
            data=st.session_state.export_data,
            file_name="rag_chat_export.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            "✖ Remove Export",
            use_container_width=True
        ):

            st.session_state.export_data = None

            st.rerun()

    st.divider()

    st.caption("Backend")

    st.code(API_URL)


# ==========================================
# MAIN TITLE
# ==========================================

st.title("🤖 Advanced RAG Assistant")

st.caption(
    "Ask questions from your document knowledge base."
)


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # ----------------------------------
        # Display sources
        # ----------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Retrieved Sources"
            ):

                for i, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Source {i}**"
                    )

                    if isinstance(source, dict):

                        if "source" in source:

                            st.write(
                                f"**File:** "
                                f"{source['source']}"
                            )

                        if "page" in source:

                            st.write(
                                f"**Page:** "
                                f"{source['page']}"
                            )

                        if "content" in source:

                            st.write(
                                source["content"]
                            )

                        else:

                            st.json(source)

                    else:

                        st.write(source)

                    st.divider()


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ==========================================
# PROCESS QUESTION
# ==========================================

if question:

    # --------------------------------------
    # Display user message
    # --------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # --------------------------------------
    # Call FastAPI
    # --------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Retrieving information and generating answer..."
        ):

            try:

                start_time = time.time()
                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=120
                )
                request_time = time.time() - start_time
                print(f"Request time: {request_time:.2f} seconds")
                response.raise_for_status()

                result = response.json()

                # ==================================
                # EXTRACT ANSWER
                # ==================================

                if isinstance(result, dict):

                    if "answer" in result:

                        answer = result["answer"]

                    elif "response" in result:

                        answer = result["response"]

                    else:

                        answer = str(result)

                else:

                    answer = str(result)

                # ==================================
                # EXTRACT SOURCES
                # ==================================

                sources = []

                if isinstance(result, dict):

                    if "sources" in result:

                        sources = result["sources"]

                    elif "context" in result:

                        sources = result["context"]

                    elif "documents" in result:

                        sources = result["documents"]

                # ==================================
                # DISPLAY ANSWER
                # ==================================

                st.markdown(answer)

                # ==================================
                # DISPLAY SOURCES
                # ==================================

                if sources:

                    with st.expander(
                        "📚 Retrieved Sources"
                    ):

                        for i, source in enumerate(
                            sources,
                            start=1
                        ):

                            st.markdown(
                                f"### Source {i}"
                            )

                            if isinstance(
                                source,
                                dict
                            ):

                                if "source" in source:

                                    st.write(
                                        f"**File:** "
                                        f"{source['source']}"
                                    )

                                if "page" in source:

                                    st.write(
                                        f"**Page:** "
                                        f"{source['page']}"
                                    )

                                if "content" in source:

                                    st.write(
                                        source["content"]
                                    )

                                else:

                                    st.json(source)

                            else:

                                st.write(source)

                            st.divider()

                # ==================================
                # SAVE ASSISTANT MESSAGE
                # ==================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            # ==================================
            # ERROR HANDLING
            # ==================================

            except requests.exceptions.ConnectionError:

                error_message = (
                    "❌ Could not connect to FastAPI. "
                    "Make sure the FastAPI server is running."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )

            except requests.exceptions.Timeout:

                error_message = (
                    "⏱️ The request timed out."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )

            except requests.exceptions.HTTPError as e:

                error_message = (
                    f"❌ FastAPI HTTP error: {e}"
                )

                st.error(error_message)

                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )

            except Exception as e:

                error_message = (
                    f"❌ Unexpected error: {e}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )