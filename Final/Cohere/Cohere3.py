import json
import cohere
import streamlit as st
import time
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
# h48tGjM4yu3P2Ei7cQKksIItpu72fQNl4saDorZe
# epOVT4qDQZjw2fmUxFts1ilaOyivjIOO8AqocChT
# Initialize Cohere
co = cohere.Client("h48tGjM4yu3P2Ei7cQKksIItpu72fQNl4saDorZe")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load and parse the JSON file
def load_menu_json(file_path):
    try:
        with open(file_path, 'r') as file:
            menu_data = json.load(file)
        return menu_data
    except json.JSONDecodeError:
        st.error("Error loading the JSON file. Please ensure the file is valid.")
        return None

# Function to create a FAISS index
def create_faiss_index(menu_data):
    menu_items = menu_data["menu"]
    item_descriptions = [f"{item['itemName']}: {item['description']}" for item in menu_items]
    embeddings = embedding_model.encode(item_descriptions)
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, menu_items

# Function to retrieve relevant menu items
def retrieve_relevant_menu_items(query, faiss_index, menu_items, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    relevant_items = [menu_items[idx] for idx in indices[0] if idx < len(menu_items)]
    return relevant_items

# Function to generate an answer using Cohere
def get_answer(query, faiss_index, menu_items, chat_history, current_item=None):
    # Use the current item if the query is contextually related
    if current_item and ("it" in query.lower() or "this" in query.lower()):
        query = f"{current_item['itemName']}: {query}"
    
    # Retrieve relevant items
    relevant_items = retrieve_relevant_menu_items(query, faiss_index, menu_items)
    
    # If an item is found, set it as the current item
    current_item = relevant_items[0] if relevant_items else None
    
    # Create a contextually aware prompt
    context = "\n".join([f"Q: {item['query']}\nA: {item['answer']}" for item in chat_history])
    prompt = (
        f"{context}\n\n"
        f"Answer the following question based on this menu by using the appropriate fields with respect to the input:\n\n"
        f"{json.dumps({'menu': relevant_items})}\n\n"
        f"Question: {query}\n\n"
        # f"I want you to respond like a waiter.\n\n"
        f"If the required item is not in the given data, you can inform the user that it is not available."
        f"Do not provide description unless it is asked."
        f"If the user asked to provide list, then display the entire list of the appropriate with only the item names."
        # f"Every time you fetch an answer, fetch the respective itemId of that respective dish and mention it seperately as a bulletin point"
    )

    # prompt = (
    #         f"{context}\n\n"
    #         f"Answer the following question based on the provided menu data. Use only the relevant fields from the input to construct your response:\n\n"
    #         f"{json.dumps({'menu': relevant_items}, indent=2)}\n\n"
    #         f"Question: {query}\n\n"
    #         f"Guidelines for your response:\n"
    #         f"- Respond as a professional waiter.\n"
    #         f"- If the requested item is not in the menu, inform the user politely that it is unavailable.\n"
    #         f"- Do not include descriptions of items unless explicitly requested by the user.\n"
    #         f"- If the user asks for a list, display only the item names from the relevant section, without additional details.\n"
    #         # f"- Whenever you refer to a specific dish, also mention its respective `itemId` as a separate bullet point.\n\n"
    #         f"Ensure your response is clear, concise, and formatted in a friendly and formal tone."
    #     )
    try:
        
        # Measure the start time
        start_time = time.time()
        # Use Cohere LLM to generate an answer
        response = co.generate(
            model='command-r-plus-08-2024',
            prompt=prompt,
            max_tokens=150,
            temperature=0.8,
            k=0,
            p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        # Measure the end time
        end_time = time.time()
        # Calculate the execution time
        execution_time = end_time - start_time
        # Extract and return the answer along with the execution time
        answer = response.generations[0].text.strip()
        return answer, execution_time, current_item
    except Exception as e:  # Catch all exceptions
        st.error(f"Error generating answer with Cohere: {e}")
        return "Failed to generate answer using Cohere API.", 0, current_item

# Load menu JSON file and create FAISS index
menu_data = load_menu_json("menudata.json")
if menu_data:
    faiss_index, menu_items = create_faiss_index(menu_data)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_item" not in st.session_state:
    st.session_state.current_item = None

# Streamlit UI
st.title("MaghilMenu")

# Input text box for user query
query = st.text_input("Ask a question about the menu:")

# Submit button
if st.button("Submit"):
    if query:
        # Generate answer and capture execution time
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

# Display the session history with the latest message on top
st.subheader("Chat History")
for chat in st.session_state.chat_history:
    st.markdown(f"**Q:** {chat['query']}")
    st.markdown(f"**A:** {chat['answer']}")
    st.markdown(f"**Execution Time:** {chat['execution_time']:.2f} seconds")
    st.markdown("---")