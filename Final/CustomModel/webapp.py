import streamlit as st
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import process

# Load intent configuration
with open('Intent.json', 'r') as f:
    intents_config = json.load(f)['intents']

# Load menu data
with open('mockMenu 1.json', 'r') as f:
    menu_data = json.load(f)['menu']

# Prepare the corpus for BM25 by combining 'itemName' and 'description' fields
corpus = [
    (item['itemName'] + " " + (item.get('description', '') if item.get('description') else ""))
    for item in menu_data
]
bm25 = BM25Okapi([doc.split() for doc in corpus])

# Prepare dish names for keyword extraction
dish_names = [item['itemName'] for item in menu_data]

# Load Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Data preprocessing for intent classification
text_input = []
intents = []
response_for_intent = {}
query_for_intent = {}

for intent in intents_config:
    intent_name = intent['intent']
    for text in intent['text']:
        text_input.append(text)
        intents.append(intent_name)
    response_for_intent[intent_name] = intent['responses'][0]  # Get the first response template
    query_for_intent[intent_name] = intent.get('query', '')

# Tokenizer and data preparation
tokenizer = Tokenizer(filters='', oov_token='<unk>')
tokenizer.fit_on_texts(text_input)
sequences = tokenizer.texts_to_sequences(text_input)
padded_sequences = pad_sequences(sequences, padding='pre')

# Prepare categorical target
intent_to_index = {intent: index for index, intent in enumerate(set(intents))}
index_to_intent = {v: k for k, v in intent_to_index.items()}  # reverse map for predicted intent
categorical_target = [intent_to_index[intent] for intent in intents]

# One-hot encoding
categorical_vec = tf.keras.utils.to_categorical(categorical_target)

# Define the model
embed_dim = 300
lstm_num = 50
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, embed_dim),
    Bidirectional(LSTM(lstm_num, dropout=0.1)),
    Dense(lstm_num, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    Dense(categorical_vec.shape[1], activation='softmax')
])

# Compile and fit the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, categorical_vec, epochs=100, verbose=1)

# Predict intent function
def predict_intent(sentence):
    tokens = [tokenizer.word_index.get(word, tokenizer.word_index['<unk>']) for word in sentence.split()]
    sent_tokens = pad_sequences([tokens], padding='pre', maxlen=padded_sequences.shape[1])  # Shape: (1, sequence_length)
    
    # No need to use np.expand_dims, as pad_sequences already returns the correct shape
    sent_tokens = np.array(sent_tokens)  # Shape: (1, sequence_length)
    
    # Predict intent
    pred = model.predict(sent_tokens)  # Model expects shape (1, sequence_length)
    pred_class = np.argmax(pred, axis=1)[0]
    
    return index_to_intent[pred_class]

def extract_dish_name(query):
    query = query.lower()
    best_match = process.extractOne(query, dish_names)
    if best_match[1] > 85:  # You can set a threshold for matching score
        return best_match[0]
    return None

# Retrieve document function with BM25 only
def retrieve_document(query):
    dish_name = extract_dish_name(query)
    if not dish_name:
        return None
    # Get BM25 scores
    tokenized_query = dish_name.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_doc_index = np.argmax(doc_scores)
    
    return menu_data[top_doc_index] if doc_scores[top_doc_index] > 0 else None

# Functions to retrieve specific data from the menu
def retrieve_dish_description(query):
    item = retrieve_document(query)
    if item:
        return item.get('itemName', 'Unknown dish'), item.get('description', 'No description available')
    return None, None

def retrieve_dish_allergic_info(query):
    item = retrieve_document(query)
    if item:
        return item.get('itemName', 'Unknown dish'), item.get('allergicInfo', 'No allergic info available')
    return None, None

def retrieve_dish_price(query):
    item = retrieve_document(query)
    if item:
        return item.get('itemName', 'Unknown dish'), item.get('price', 0)
    return None, None

def get_kids_friendly_dishes():
    return [item for item in menu_data if item.get('kidsFriendly')]

def get_spicy_dishes_for_fever():
    return [item for item in menu_data if any(f.get('name', '').lower() == 'spicy' for f in item.get('itemFilter', []))]

def get_vegan_dishes():
    return [item for item in menu_data if any(f.get('name', '').lower() == 'vegan' for f in item.get('itemFilter', []))]

def get_nut_free_dishes():
    return [item for item in menu_data if any(f.get('name', '').lower() == 'nut-free' for f in item.get('itemFilter', []))]

def get_fish_free_dishes():
    return [item for item in menu_data if 'fish' not in item.get('allergicInfo', '').lower()]

# def find_dish_with_least_prep_time():
#     sorted_dishes = sorted(menu_data, key=lambda x: x.get('prepTime', float('inf')))
#     return sorted_dishes[:10]  # Return only the top 10 items with the least prep time

def find_dish_with_least_prep_time():
    # Sort the dishes by prep time and limit to top 10
    sorted_dishes = sorted(menu_data, key=lambda x: x.get('prepTime', float('inf')))
    
    # Limit the result to 10 items
    top_10_dishes = sorted_dishes[:10]
    
    # Print the names of the top 10 dishes with the least prep time
    for dish in top_10_dishes:
        print(dish.get('itemName', 'Unknown dish'))

    return top_10_dishes 


def rag_operation(intent, query):
    response_template = response_for_intent.get(intent, "I'm not sure how to respond.")
    dishes = ""

    # Define a function mapping
    intent_functions = {
        "GetKidsFriendlyDishes": get_kids_friendly_dishes,
        "GetSpicyDishesForFever": get_spicy_dishes_for_fever,
        "GetVeganDishes": get_vegan_dishes,
        "GetNutFreeDishes": get_nut_free_dishes,
        "GetFishFreeDishes": get_fish_free_dishes,
        "FindDishWithLeastPrepTime": find_dish_with_least_prep_time,
        "RetrieveDishDescription": retrieve_dish_description,
        "RetrieveDishAllergicInfo": retrieve_dish_allergic_info,
        "RetrieveDishPrice": retrieve_dish_price
    }

    if intent in intent_functions:
        if intent in ["RetrieveDishDescription", "RetrieveDishAllergicInfo", "RetrieveDishPrice"]:
            item_name, result = intent_functions[intent](query)
            if item_name:
                if intent == "RetrieveDishPrice":
                    dishes = response_template.format(dish_name=item_name, price=result)
                elif intent == "RetrieveDishAllergicInfo":
                    dishes = response_template.format(dish_name=item_name, allergic_info=result)
                else:  # RetrieveDishDescription
                    dishes = response_template.format(dish_name=item_name, description=result)
            else:
                dishes = "Information not available."
        else:
            # For list-based responses, limit the result to 5 dishes
            dishes_list = intent_functions[intent]()
            if dishes_list:
                dishes = ', '.join([item['itemName'] for item in dishes_list[:5]]) or "No dishes available."
                dishes = response_template.format(dishes=dishes)
            else:
                dishes = "No dishes available."
    else:
        dishes = "I'm not sure how to respond to that."

    return dishes
# Response construction: links intent, query, and RAG operation
def response(sentence):
    # Step 1: Predict intent
    intent_name = predict_intent(sentence)
    
    # Step 2: Perform RAG (Retrieve Augment Generate) operation
    rag_output = rag_operation(intent_name, sentence)

    return rag_output, intent_name

# Streamlit app interface
st.title("Restaurant Chatbot")
st.subheader("Ask me about our menu!")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Input box for user query
user_input = st.text_input("You:", "")

# Send button
if st.button("Send"):
    if user_input:
        bot_response, intent_name = response(user_input)
        # Append user input and bot response to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

# Display chat history
for chat in st.session_state.chat_history:
    st.write(f"You: {chat['user']}")
    st.write(f"Bot: {chat['bot']}")