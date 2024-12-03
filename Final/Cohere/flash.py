#API: AIzaSyDikCFTjz3PjfxcHueRKAfrMY7y2Ft6tJM
#Model: gemini-1.5-flash

import json
import streamlit as st
import time
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyDikCFTjz3PjfxcHueRKAfrMY7y2Ft6tJM")

# Initializing the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Parsing the JSON File
def load_menu_json(file_path):
    try:
        with open(file_path, 'r') as file:
            menu_data = json.load(file)
        return menu_data
    except json.JSONDecodeError:
        st.error("Error loading the JSON file. Please ensure the file is valid.")
        return None

# FAISS Index
def create_faiss_index(menu_data):
    menu_items = menu_data["menu"]
    item_descriptions = [f"{item['itemName']}: {item['description']}" for item in menu_items]
    embeddings = embedding_model.encode(item_descriptions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, menu_items

# Retrievial of the relevant items
def retrieve_relevant_menu_items(query, faiss_index, menu_items, top_k=None):
    if top_k is None:
        top_k = len(menu_items)  # Default to retrieving all items
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    relevant_items = [menu_items[idx] for idx in indices[0] if idx < len(menu_items)]
    return relevant_items

# Answer Generation using Gemini API
def get_answer(query, faiss_index, menu_items, chat_history, current_item=None):
    # Adjust top_k dynamically based on query context
    top_k = len(menu_items) if "all" in query.lower() or "list" in query.lower() else 5
    
    # Retrieve relevant items
    relevant_items = retrieve_relevant_menu_items(query, faiss_index, menu_items, top_k)
    
    # Contextual information
    if current_item and ("it" in query.lower() or "this" in query.lower() or "that" in query.lower()):
        query = f"{current_item['itemName']}: {query}"
    current_item = relevant_items[0] if relevant_items else None

    # Prompt 
    context = "\n".join([f"Q: {item['query']}\nA: {item['answer']}" for item in chat_history])
    prompt = (
        f"{context}\n\n"
        f"Answer the following question based on this menu by using the appropriate fields with respect to the input:\n\n"
        f"{json.dumps({'menu': relevant_items}, indent=2)}\n\n"
        f"Question: {query}\n\n"
        f"Response Guidelines:\n"
        f"- Respond as a professional waiter.\n"
        f"- If the requested item is not in the menu, inform the user politely that it is unavailable.\n"
        f"- Do not include descriptions of items unless explicitly requested by the user.\n"
        f"- If the user asks for a list, display only the item names from the relevant section, without additional details.\n"

    )

    try:
        # Measuring the start time
        start_time = time.time()

        # Using Google Generative AI to generate an answer
        model = genai.GenerativeModel("gemini-1.5-flash")  # Initialize the model
        response = model.generate_content(prompt)
        
        # Measuring the end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Extracting and returning the answer along with execution time
        answer = response.text.strip()
        return answer, execution_time, current_item
    except Exception as e:
        st.error(f"Error generating answer with Gemini: {e}")
        return "Failed to generate answer using Gemini API.", 0, current_item

# Loading menu data and creating FAISS index
menu_data = load_menu_json("menudata.json")
if menu_data:
    faiss_index, menu_items = create_faiss_index(menu_data)

# Initializing session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_item" not in st.session_state:
    st.session_state.current_item = None

# Streamlit UI
st.title("MaghilMenu")
query = st.text_input("Ask a question about the menu:")
if st.button("Submit"):
    if query:
        # Answer generation and capturing the execution time
        answer, execution_time, current_item = get_answer(
            query, faiss_index, menu_items, st.session_state.chat_history, st.session_state.current_item
        )
        # Update session state variables
        st.session_state.chat_history.insert(0, {
            "query": query,
            "answer": answer,
            "execution_time": execution_time
        })
        st.session_state.current_item = current_item

st.subheader("Chat History")
for chat in st.session_state.chat_history:
    st.markdown(f"**Q:** {chat['query']}")
    st.markdown(f"**A:** {chat['answer']}")
    st.markdown(f"**Execution Time:** {chat['execution_time']:.2f} seconds")
    st.markdown("---")