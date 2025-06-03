# streamlit.py
import streamlit as st
import pickle
import numpy as np
from preprocess import preprocess  # Assuming you'll save the preprocessing function in a separate file

# Load the model and vectorizer
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('cv.pkl', 'rb'))
    return model, cv

model, cv = load_artifacts()

# Define the feature extraction functions (copied from your original code)
def test_common_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def test_total_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))

def test_fetch_token_features(q1, q2):
    from nltk.corpus import stopwords
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    token_features = [0.0]*8
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

def test_fetch_length_features(q1, q2):
    import distance
    length_features = [0.0]*3
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features

def test_fetch_fuzzy_features(q1, q2):
    from fuzzywuzzy import fuzz
    fuzzy_features = [0.0]*4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features

def query_point_creator(q1, q2):
    input_query = []
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2)/test_total_words(q1, q2), 2))

    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))

# Streamlit UI
st.title("Question Similarity Detector")
st.write("This app predicts whether two questions are similar (duplicates) or not.")

q1 = st.text_area("Enter Question 1:", "Where is the capital of India?")
q2 = st.text_area("Enter Question 2:", "What is the current capital of Pakistan?")

if st.button("Check Similarity"):
    # Create the feature vector
    features = query_point_creator(q1, q2)
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    st.subheader("Results")
    if prediction[0] == 1:
        st.success("The questions are similar (duplicates)")
    else:
        st.error("The questions are not similar")
    
    st.write(f"Probability of being similar: {probability[0][1]:.2f}")
    st.write(f"Probability of being dissimilar: {probability[0][0]:.2f}")