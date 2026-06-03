import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()

st.set_page_config(
    page_title="Railway Passenger Assistant",
    page_icon="🚆",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>

.main {
    background-color: #0b1220;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

.title {
    text-align: center;
    padding: 10px;
}

.big-title {
    font-size: 40px;
    font-weight: bold;
    color: white;
}

.subtitle {
    color: #b0b0b0;
    font-size: 18px;
}

.metric-box {
    background-color: #162033;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.source-box {
    background-color: #18263f;
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:

    st.title("🚆 Railway RAG")

    st.markdown("---")

    st.subheader("📌 About")

    st.write("""
    AI-powered Railway Passenger Assistant

    Features:
    - Railway Rules
    - Refund Information
    - Tatkal Rules
    - Luggage Rules
    - RAG + FAISS
    - Groq Llama Model
    """)

    st.markdown("---")

    st.subheader("📊 System Status")

    st.success("Vector DB Loaded")
    st.success("Groq Connected")
    st.success("RAG Active")

    st.markdown("---")

    st.subheader("💡 Suggested Questions")

    st.caption("Try asking:")

    st.write("• What is Tatkal booking?")
    st.write("• Can I get refund on Tatkal ticket?")
    st.write("• How much luggage can I carry?")
    st.write("• Can I carry pets in train?")

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div class='title'>
<div class='big-title'>🚆 Railway Passenger Information Assistant</div>
<div class='subtitle'>
AI Powered Railway Rules & Passenger Guidance Chatbot
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ---------------------------
# METRICS
# ---------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Documents", "4")

with col2:
    st.metric("Vector DB", "FAISS")

with col3:
    st.metric("Model", "Llama 3.1")

with col4:
    st.metric("Status", "Active")

st.markdown("---")

# ---------------------------
# LOAD EMBEDDINGS + DB
# ---------------------------
@st.cache_resource
def load_vector_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

db = load_vector_db()

# ---------------------------
# CHAT HISTORY
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------
# CHAT INPUT
# ---------------------------
prompt = st.chat_input(
    "Ask any Railway related question..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("🔍 Searching Railway Documents..."):

            docs = db.similarity_search(
                prompt,
                k=3
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            rag_prompt = f"""
You are a Railway Passenger Assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
say:

"I could not find relevant information in the railway documents."

Context:
{context}

Question:
{prompt}
"""

            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY")
            )

            response = llm.invoke(rag_prompt)

            answer = response.content

            st.markdown(answer)

            # Sources
            with st.expander("📄 Retrieved Context"):

                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"### Source {i}")
                    st.write(doc.page_content[:1000])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )