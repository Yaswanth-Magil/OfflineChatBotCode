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
with open('Intent 2.json', 'r') as f:
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

    # Predict intent
    pred = model.predict(sent_tokens)  # Model expects shape (1, sequence_length)
    pred_class = np.argmax(pred, axis=1)[0]

    return index_to_intent[pred_class]

def extract_dish_name(query):
    query = query.lower()
    best_match = process.extractOne(query, dish_names)
    if best_match[1] > 85:  # Set a threshold for matching score
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
    spicy_keyword = 'spicy'
    spicy_dishes = []
    for index, item in enumerate(menu_data):
        if not isinstance(item, dict):
            print(f"Warning: Item at index {index} is not a dictionary. Skipping.")
            continue

        description = item.get('description', '')
        if not isinstance(description, str):
            description = str(description) if description is not None else ''
        description = description.lower()

        item_filters = item.get('itemFilter', [])
        if not isinstance(item_filters, list):
            item_filters = []
        item_filters = [str(f.get('name', '')).lower() for f in item_filters if isinstance(f, dict)]

        if spicy_keyword in description or spicy_keyword in item_filters:
            spicy_dishes.append(item)
            if len(spicy_dishes) >= 5:
                break  # Limit to 5 dishes

    return spicy_dishes

def get_vegan_dishes():
    return [item for item in menu_data if any(f.get('name', '').lower() == 'vegan' for f in item.get('itemFilter', []))]

def get_nut_free_dishes():
    nut_keywords = ['nut', 'nuts', 'peanut', 'almond', 'walnut', 'cashew', 'hazelnut']
    nut_free_dishes = []
    for index, item in enumerate(menu_data):
        if not isinstance(item, dict):
            print(f"Warning: Item at index {index} is not a dictionary. Skipping.")
            continue

        allergic_info = item.get('allergicInfo', '')
        if not isinstance(allergic_info, str):
            allergic_info = str(allergic_info) if allergic_info is not None else ''
        allergic_info = allergic_info.lower()

        description = item.get('description', '')
        if not isinstance(description, str):
            description = str(description) if description is not None else ''
        description = description.lower()

        if not any(nut in allergic_info for nut in nut_keywords) and not any(nut in description for nut in nut_keywords):
            nut_free_dishes.append(item)
            if len(nut_free_dishes) >= 5:
                break  # Limit to 5 dishes

    return nut_free_dishes

def get_fish_free_dishes():
    fish_free_dishes = [item for item in menu_data if 'fish' not in item.get('allergicInfo', '').lower()]

    for index, item in enumerate(menu_data):
        if len(fish_free_dishes) >= 5:
            break
    
    return fish_free_dishes




def find_dish_with_least_prep_time():
    sorted_dishes = sorted(menu_data, key=lambda x: x.get('prepTime', float('inf')))
    return sorted_dishes[:10]  # Return only the top 10 items with the least prep time

def find_recommended_items(item, menu):
    category = item.get('category', '')
    recommendations = [menu_item for menu_item in menu if menu_item.get('category') == category and menu_item['itemName'] != item['itemName']]
    return recommendations[:3]  # Recommend up to 3 items

def recommend_dishes_for_pairing(dish_name):
    item = retrieve_document(dish_name)
    if item:
        return find_recommended_items(item, menu_data)
    return []

def rag_operation(intent, query):
    response_template = response_for_intent.get(intent, "I'm not sure how to respond.")
    dishes = ""
    dish_name = ""
    recommended_dishes = ""

    intent_functions = {
        "GetKidsFriendlyDishes": get_kids_friendly_dishes,
        "GetSpicyDishesForFever": get_spicy_dishes_for_fever,
        "GetVeganDishes": get_vegan_dishes,
        "GetNutFreeDishes": get_nut_free_dishes,
        "GetFishFreeDishes": get_fish_free_dishes,
        "FindDishWithLeastPrepTime": find_dish_with_least_prep_time,
    }

    if intent in intent_functions:
        items = intent_functions[intent]()
        dishes = ', '.join([item['itemName'] for item in items]) if items else "No dishes found."
    elif intent == "RecommendDishesForPairing":
        # Fetch dish name from query and recommendations
        dish_name = extract_dish_name(query)
        recommendations = recommend_dishes_for_pairing(dish_name)
        
        # Format the recommended dishes
        recommended_dishes = ', '.join([rec['itemName'] for rec in recommendations]) if recommendations else "No recommendations found."
    else:
        if intent in ["RetrieveDishDescription", "RetrieveDishAllergicInfo", "RetrieveDishPrice"]:
            item_name, result = None, None
            if intent == "RetrieveDishDescription":
                item_name, result = retrieve_dish_description(query)
            elif intent == "RetrieveDishAllergicInfo":
                item_name, result = retrieve_dish_allergic_info(query)
            elif intent == "RetrieveDishPrice":
                item_name, result = retrieve_dish_price(query)

            if item_name and result:
                if intent == "RetrieveDishDescription":
                    dishes = f"{item_name}: {result}"
                elif intent == "RetrieveDishAllergicInfo":
                    dishes = f"{item_name}: {result}"
                elif intent == "RetrieveDishPrice":
                    dishes = f"The price of {item_name} is ${result}."
                
                # Get recommendations based on the item
                recommendations = recommend_dishes_for_pairing(item_name)
                if recommendations:
                    rec_names = ', '.join([rec['itemName'] for rec in recommendations])
                    dishes += f"\nYou may also like: {rec_names}."
            else:
                dishes = "Dish not found."
        else:
            items = intent_functions[intent]()
            dishes = ', '.join([item['itemName'] for item in items]) if items else "No dishes found."

    # Replace placeholders in the response
    response = response_template.replace("{dishes}", dishes)
    response = response.replace("{dish_name}", dish_name if dish_name else "this dish")
    response = response.replace("{recommended_dishes}", recommended_dishes)

    return response


# Streamlit interface with session history and submit button
st.title("Restaurant Chatbot")

# Initialize session state if not already
if "history" not in st.session_state:
    st.session_state.history = []

# Input field for user query
user_input = st.text_input("Ask me about our menu:")

# Submit button to trigger intent prediction and response generation
if st.button("Submit"):
    if user_input:
        intent = predict_intent(user_input)
        response = rag_operation(intent, user_input)
        st.session_state.history.append({"input": user_input, "response": response})  # Add new input/response to history

# Display conversation history, newest at the top
if st.session_state.history:
    for chat in reversed(st.session_state.history):
        st.write(f"**You:** {chat['input']}")
        st.write(f"**Bot:** {chat['response']}")
