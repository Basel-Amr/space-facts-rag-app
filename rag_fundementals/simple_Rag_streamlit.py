# 🚀 Space Facts RAG System

import streamlit as st
import csv
import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

# ========== 🔧 ENVIRONMENT SETUP ==========
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# ========== 📦 MODELS ==========
class EmbeddingModel:
    def __init__(self, model_type="openai"):
        """
        Initialize the embedding model based on the selected type.
        """
        self.model_type = model_type

        if model_type == "openai":
            self.client = OpenAI(api_key=API_KEY)
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=API_KEY, model_name="text-embedding-ada-002"
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )


class LLMModel:
    def __init__(self, model_type="openai"):
        """
        Initializes the language model (LLM) based on the selected type.
        """
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=API_KEY)
            self.model_name = "gpt-4o-mini"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"


# ========== 📄 UTILITY FUNCTIONS ==========
def generate_csv():
    """Creates a CSV file with predefined space facts."""
    facts = [
        {"id": 1, "fact": "The first human to orbit Earth was Yuri Gagarin in 1961."},
        {"id": 2, "fact": "Apollo 11 landed the first humans on the Moon in 1969."},
        {"id": 3, "fact": "The Hubble Space Telescope was launched in 1990."},
        {"id": 4, "fact": "Mars is the most explored planet in the solar system."},
        {"id": 5, "fact": "The ISS has been continuously occupied since November 2000."},
        {"id": 6, "fact": "Voyager 1 is the farthest human-made object from Earth."},
        {"id": 7, "fact": "SpaceX is the first private company to send humans to orbit."},
        {"id": 8, "fact": "The James Webb Telescope is the successor to Hubble."},
        {"id": 9, "fact": "The Milky Way contains over 100 billion stars."},
        {"id": 10, "fact": "Black holes are regions where gravity prevents escape."},
    ]
    with open("space_facts.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)
    return facts


def setup_chromadb(documents, embedding_model):
    """Sets up ChromaDB collection."""
    client = chromadb.Client()
    try:
        client.delete_collection("space_facts")
    except:
        pass

    collection = client.create_collection(
        name="space_facts", embedding_function=embedding_model.embedding_fn
    )
    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])
    return collection


def find_related_chunks(query, collection, top_k=2):
    results = collection.query(query_texts=[query], n_results=top_k)
    return list(zip(results["documents"][0], results.get("metadatas", [{}])))


def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def rag_pipeline(query, collection, llm_model, top_k=2):
    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)
    response = llm_model.generate_completion([
        {"role": "system", "content": "Answer only using the given context."},
        {"role": "user", "content": augmented_prompt},
    ])
    references = [chunk[0] for chunk in related_chunks]
    return response, references, augmented_prompt


# ========== 🎨 STREAMLIT APP ==========
def streamlit_app():
    st.set_page_config(page_title="🚀 Space Facts RAG", layout="wide")

    # ✅ Custom Header with Logo & Title
    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <div style='display: flex; align-items: center;'>
                <img src='https://cdn-icons-png.flaticon.com/512/3214/3214426.png' width='60'>
                <h1 style='padding-left: 15px;'>Space Facts RAG System</h1>
            </div>
            <img src='https://media.giphy.com/media/KGhpQH1Zi2CLk/giphy.gif' width='80'>
        </div>
        <hr style="border: 1px solid #bbb;">
    """, unsafe_allow_html=True)

    # ⚙️ Sidebar: Model Configuration
    st.sidebar.markdown("### ⚙️ Model Settings")

    llm_type = st.sidebar.radio(
        "🔧 Choose LLM Model:",
        ["openai", "ollama"],
        format_func=lambda x: "🤖 OpenAI GPT-4" if x == "openai" else "🦙 Ollama LLaMA3"
    )

    embedding_type = st.sidebar.radio(
        "🔬 Choose Embedding Model:",
        ["openai", "chroma", "nomic"],
        format_func=lambda x: {
            "openai": "🔷 OpenAI Embedding",
            "chroma": "🟠 Chroma Default",
            "nomic": "🟡 Nomic (Ollama)",
        }[x]
    )

    # 🧠 Initialization
    if "initialized" not in st.session_state:
        st.session_state.facts = generate_csv()
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        st.session_state.collection = setup_chromadb(
            [fact["fact"] for fact in st.session_state.facts],
            st.session_state.embedding_model
        )
        st.session_state.initialized = True

    # 🔄 If models are changed
    if (
        st.session_state.llm_model.model_type != llm_type or
        st.session_state.embedding_model.model_type != embedding_type
    ):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        st.session_state.collection = setup_chromadb(
            [fact["fact"] for fact in st.session_state.facts],
            st.session_state.embedding_model
        )

    # 📚 Show Available Space Facts
    with st.expander("📚 Click to View Space Facts", expanded=False):
        st.markdown("<ul>" + "".join([f"<li>{fact['fact']}</li>" for fact in st.session_state.facts]) + "</ul>", unsafe_allow_html=True)

    # 🔍 Ask a Question
    query = st.text_input("🔭 Ask a question about space:", placeholder="e.g., What is the Hubble Telescope?")

    if query:
        with st.spinner("🛰️ Retrieving knowledge from the stars..."):
            response, references, prompt = rag_pipeline(
                query, st.session_state.collection, st.session_state.llm_model
            )

        # ✨ Animated Typing Effect
        animated_response = f"""
        <div style="background-color:#0d1117; color:#39ff14; padding:20px; border-radius:10px; font-family: monospace; font-size: 16px;">
            <strong>🤖 AI Response:</strong><br><br>
            <span class="typewriter">{response}</span>
        </div>
        <style>
        .typewriter {{
            overflow: hidden;
            border-right: .15em solid orange;
            white-space: nowrap;
            animation: typing 3.5s steps(40, end), blink-caret .75s step-end infinite;
        }}

        @keyframes typing {{
            from {{ width: 0 }}
            to {{ width: 100% }}
        }}

        @keyframes blink-caret {{
            from, to {{ border-color: transparent }}
            50% {{ border-color: orange }}
        }}
        </style>
        """
        st.markdown(animated_response, unsafe_allow_html=True)

        # 📖 References Used
        st.markdown("### 📎 Source Chunks")
        for ref in references:
            st.markdown(f"🔹 {ref}")

        # 🧪 Technical Info (Expandable)
        with st.expander("🧪 Technical Info & Prompt Debug"):
            st.markdown("#### 🧠 Augmented Prompt")
            st.code(prompt)

            st.markdown("#### ⚙️ Model Configuration")
            st.markdown(f"- **LLM:** `{llm_type.upper()}`")
            st.markdown(f"- **Embedding:** `{embedding_type.upper()}`")


    st.markdown("""
        <hr>
        <div style='text-align: center;'>
            <p>🚀 Built with ❤️ by <strong>Basel Amr Barakat</strong></p>
            <p>
                <a href='https://www.linkedin.com/in/baselamrbarakat' target='_blank'>
                    <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='25' style='vertical-align:middle;'/> LinkedIn
                </a>
                &nbsp;&nbsp;&nbsp;
                <a href='https://github.com/Basel-Amr' target='_blank'>
                    <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='25' style='vertical-align:middle;'/> GitHub
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)




# ========== 🚀 MAIN ==========
if __name__ == "__main__":
    streamlit_app()
