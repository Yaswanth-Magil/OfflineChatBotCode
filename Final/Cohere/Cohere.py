import json
import cohere
import streamlit as st
import time  # Import the time module

# Initialize Cohere
co = cohere.Client("epOVT4qDQZjw2fmUxFts1ilaOyivjIOO8AqocChT")

# Function to load and parse the JSON file
def load_menu_json(file_path):
    try:
        with open(file_path, 'r') as file:
            menu_data = json.load(file)
        return menu_data
    except json.JSONDecodeError:
        st.error("Error loading the JSON file. Please ensure the file is valid.")
        return None

# Function to limit the number of menu items in the prompt
def limit_menu_data(menu_data, max_items=150):
    # Limit the number of items in the menu to avoid exceeding token limits
    return menu_data["menu"][:max_items]

# Function to retrieve and generate an answer from menu data with contextual history
def get_answer(query, menu_data, chat_history):
    # Limit the data to avoid too many tokens
    limited_menu_data = limit_menu_data(menu_data)

    # Create a contextually aware prompt by including the chat history in the context
    context = "\n".join([f"Q: {item['query']}\nA: {item['answer']}" for item in chat_history])
    prompt = f"{context}\n\nAnswer with in 150 words the following question based on this menu by using the appropriate fields with respect to the input:\n\n{json.dumps({'menu': limited_menu_data})}\n\nQuestion: {query}\n\n I want you to respond like a waiter\n\n If The required item is not in the given data, you can inform the user that it is not available."

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
        return answer, execution_time
    except cohere.exceptions.CohereError as e:
        st.error(f"Error generating answer with Cohere: {e}")
        return "Failed to generate answer using Cohere API.", 0

# Load menu JSON file once for the session
menu_data = load_menu_json("mockMenu 1.json")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("MaghilMenu")

# Input text box for user query
query = st.text_input("Ask a question about the menu:")

# Submit button
if st.button("Submit"):
    if query:
        # Generate answer and capture execution time
        answer, execution_time = get_answer(query, menu_data, st.session_state.chat_history)

        # Update chat history (newest message on top)
        st.session_state.chat_history.insert(0, {
            "query": query,
            "answer": answer,
            "execution_time": execution_time
        })

# Display the session history with the latest message on top
st.subheader("Chat History")
for chat in st.session_state.chat_history:
    st.markdown(f"**Q:** {chat['query']}")
    st.markdown(f"**A:** {chat['answer']}")
    st.markdown(f"**Execution Time:** {chat['execution_time']:.2f} seconds")
    st.markdown("---")
