import csv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Union

# Step One: Load all environment variables
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
class EmbeddingModel:
    def __init__(self, model_type= "openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.embedding_fn = embedding_functions.OpenAIEmbedding(
                api_key = api_key,
                model_name = "text-embedding-3-small"
            )
        elif model_type ==  "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            # Using Ollama nomic-emeb-text model
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key = "ollama", 
                api_base = "http://localhost:11434/v1",
                model_name = "nomic-embed-text", 
            )
            
            
class LLMModel:
    def __init__(self, model_type: str = "openai", api_key: str = None):
        """
        Initializes the LLMModel with the specified model type.

        :param model_type: Either 'openai' for OpenAI's API or 'ollama' for local inference.
        :param api_key: Required if using OpenAI. Ignored if using local (Ollama).
        """
        self.model_type = model_type.lower()
        
        if self.model_type == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI model.")
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"
        elif self.model_type == "ollama":
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2:latest"
        else:
            raise ValueError("Unsupported model_type. Choose either 'openai' or 'ollama'.")

    def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Union[str, None]:
        """
        Generates a completion from the language model based on the provided messages.

        :param messages: A list of dictionaries with 'role' and 'content' keys.
        :param temperature: Controls randomness. 0.0 is deterministic.
        :return: The generated response as a string, or an error message.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return None
            
def select_models():
    # Select LLM Model
    print("\n Select LLM Model")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama2")
    while True:
        choice = input("Enter Choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice=="1" else "ollama"
            break
        print("please enter either 1 or 2")
    
    # Selecting Embedding Model
    print("\nSelect Embedding Model")
    print("1. OpenAI Embedding")
    print("2. Chroma Default")
    print("3. Nomic Embed Text (Ollama)")
    while True:
        choice =input("Enter Choice (1, 2 or 3): ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "chroma", "3": "nomic"}[choice]
            break
        print("Please enter (1, 2 or 3)")
    return llm_type, embedding_type
    
def generate_csv():
    """
    Generates a CSV file named 'space_facts.csv' containing a collection of space-related facts.
    """
    space_facts = [
        {"id": 1, "fact": "The Hubble Space Telescope, launched in 1990, orbits Earth and captures high-resolution images of deep space, aiding major astronomical discoveries."},
        {"id": 2, "fact": "Mars exploration involves numerous missions, such as NASA's Perseverance rover, which searches for signs of ancient life and collects samples for return to Earth."},
        {"id": 3, "fact": "The Sun makes up 99.8% of the mass in our solar system and is primarily composed of hydrogen and helium."},
        {"id": 4, "fact": "Jupiter is the largest planet in our solar system and has a giant storm called the Great Red Spot, which is larger than Earth."},
        {"id": 5, "fact": "Saturn's rings are made up of billions of ice particles, ranging in size from tiny grains to chunks as big as houses."},
        {"id": 6, "fact": "Black holes are regions of spacetime where gravity is so strong that nothing—not even light—can escape from them."},
        {"id": 7, "fact": "The Milky Way galaxy is about 100,000 light-years in diameter and contains over 100 billion stars."},
        {"id": 8, "fact": "Neil Armstrong became the first human to walk on the Moon on July 20, 1969, during NASA's Apollo 11 mission."},
        {"id": 9, "fact": "Venus has a thick, toxic atmosphere primarily composed of carbon dioxide, causing a runaway greenhouse effect and extreme surface temperatures."},
        {"id": 10, "fact": "The International Space Station (ISS) is a habitable satellite that orbits Earth every 90 minutes and hosts international crew members."},
        {"id": 11, "fact": "The Moon influences Earth's tides through gravitational interaction and is slowly drifting away at a rate of about 3.8 cm per year."},
        {"id": 12, "fact": "Exoplanets are planets outside our solar system, and thousands have been discovered, some potentially habitable, using telescopes like Kepler and TESS."}
    ]

    # Write the data to CSV
    with open("space_facts.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(space_facts)

    print("✅ CSV file 'space_facts.csv' created successfully with space-related facts.")

def load_csv(file_path='space_facts.csv'):
    df = pd.read_csv(file_path)
    documents = df["fact"].tolist()
    print("\nLoaded documents")
    for doc in documents:
        print(f"- {doc}")
    return documents

def setup_chromaDB(documents, embedding_model):
    client = chromadb.Client()
    
    try:
        client.delete_collection("space_facts")
    except:
        pass
    
    collection = client.create_collection(
        name = "space_facts", 
        embedding_function = embedding_model.embedding_fn
    )
    
    collection.add(
        documents = documents,
        ids = [str(i) for i in range(len(documents))]
    )
    
    print("\nDocuments added to ChromaDB collection successfuly")
    return collection

def find_related_chunks(query, collection, top_k=2):
    results = collection.query(query_texts = [query], n_results = top_k)
    
    print("\nRelated chunks found")
    for doc in results["documents"][0]:
        print(f"_ {doc}")
        
    return list(
        zip(
            results["documents"][0],
            (
              results["metadatas"][0]
              if results["metadatas"][0]
              else [{}] * len(results["documents"][0])
            ),
        )
    )
    
def augment_prompt(query, related_chunks):
    """
    Constructs an augmented prompt by combining related context chunks and the user query.
    """
    context = "\n".join([chunk[0] for chunk in related_chunks])
    augmented_prompt = f"📄 Context:\n{context}\n\n❓ Question: {query}\n📝 Answer:"

    print("\n🧠 Augmented Prompt ↓↓↓")
    print(augmented_prompt)

    return augmented_prompt


def rag_pipeline(query, collection, llm_model, top_k = 2):
    print(f"\nProcessing Query: {query}")
    
    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)
    
    response = llm_model.generate_completion(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant who can answer questions about space but only answer questions that are directly related to the sources/documents given."
            },
            {
                "role": "user",
                "content": augmented_prompt
            }
        ]
    )
    
    print("\nGenerated response: ")
    print(response)
    
    references = [chunk[0] for chunk in related_chunks]
    return response, references

def main():
    print("Starting Rag Pipeline demo...")
    
    # Select models
    llm_type, embedding_type = select_models()
    
    # Initialize models
    llm_model = LLMModel(llm_type)
    embedding_model = EmbeddingModel(embedding_type)
    
    print(f"\nUsing LLM: {llm_type.upper()}")
    print(f"\nUsing Embeddings: {embedding_type.upper()}")
    
    # Generate and load data
    generate_csv()
    documents =load_csv()
    
    # Setup ChromaDB
    collection = setup_chromaDB(documents, embedding_model)
    
    # Run Queries
    queries = [
        "What is the Hubble Space Telescope",
        "Tell me about Mars exploration"
    ]
    
    for query in queries:
        print("\n" + "=" * 50)
        print(f"Preprocessing query: {query}")
        response, references = rag_pipeline(query, collection, llm_model)
        
        print("\nFinal Results: ")
        print("-"*30)
        print("Response:", response)
        print("\nReferences used: ")
        for ref in references:
            print(f"-{ref}")
        print("="*50)
        
if __name__ == "__main__":
    main()
    
    