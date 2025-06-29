import streamlit as st
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


@st.cache_data()
def load_model():
    data = pd.read_pickle("books.pkl")
    model = pickle.load(open("knn_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    return data, model, tfidf, scaler

data, model, tfidf, scaler = load_model()

books_title = data['Book-Title'].values
title_to_idx = {title: idx for idx, title in enumerate(books_title)}

st.title('Book Recommender System')

book_title = st.selectbox('Select a book:', books_title)

if book_title:
    idx = title_to_idx[book_title]

    text_feature = data.iloc[idx]['text']

    tfidf_feature = tfidf.transform([text_feature])
    year = scaler.transform([[data.iloc[idx]['Year-Of-Publication']]])
    combined = hstack([tfidf_feature, csr_matrix(year)])

    distances, indices = model.kneighbors(combined)

    st.header('Recommended books:')

    recommendations = []
    for i in indices[0]:
        recommendations.append(data.iloc[i]['Book-Title'])

    st.write(recommendations)