import streamlit as st
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Book Recommender", layout="wide")

st.markdown(
    """
<style>
    .header {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
    }
    .book-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .book-title {
        font-weight: bold;
        font-size: 18px;
        color: #2c3e50;
    }
    .book-author {
        font-style: italic;
        color: #7f8c8d;
    }
    .book-publisher {
        color: #3498db;
    }
    .similarity-score {
        color: #e74c3c;
        font-size: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data()
def load_model():
    try:
        data = pd.read_pickle("books.pkl")
        model = pickle.load(open("knn_model.pkl", "rb"))
        tfidf = pickle.load(open("tfidf.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))

        return data, model, tfidf, scaler

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


data, model, tfidf, scaler = load_model()

st.title("📚 Book Recommendation System")
st.markdown("Discover books similar to your favorites!")

search_term = st.text_input("Search for a book:", "")
# Filter books based on search term
if search_term:
    filtered_books = [
        title
        for title in data["Book-Title"].values
        if search_term.lower() in title.lower()
    ]
else:
    filtered_books = data["Book-Title"].values

book_title = st.selectbox("Select a book:", filtered_books)

books_titles = data["Book-Title"].values
title_to_idx = {title: idx for idx, title in enumerate(books_titles)}

if book_title:
    idx = title_to_idx[book_title]

    selected_book = data.iloc[idx]
    st.markdown(f"### You selected: {selected_book['Book-Title']}")
    st.markdown(f"**Author:** {selected_book['Book-Author']}")
    st.markdown(f"**Publisher:** {selected_book['Publisher']}")
    st.markdown(f"**Year:** {selected_book['Year-Of-Publication']}")

    # Get recommendations
    text_feature = data.iloc[idx]["text"]
    tfidf_feature = tfidf.transform([text_feature])
    year = scaler.transform([[data.iloc[idx]["Year-Of-Publication"]]])
    combined = hstack([tfidf_feature, csr_matrix(year)])
    distances, indices = model.kneighbors(combined)

    st.markdown("---")
    st.markdown("## 📖 Recommended Books")
    st.markdown(f"Books similar to *{book_title}*:")

    # Display recommendations with more info and similarity scores
    for i, (book_idx, distance) in enumerate(zip(indices[0], distances[0])):
        recommended_book = data.iloc[book_idx]

        # Convert distance to similarity score (0-100 scale)
        similarity_score = 100 * (1 - distance)

        with st.expander(f"{i+1}. {recommended_book['Book-Title']}"):
            st.markdown(
                f"""
            <div class="book-card">
                <div class="book-title">{recommended_book['Book-Title']}</div>
                <div class="book-author">by {recommended_book['Book-Author']}</div>
                <div class="book-publisher">Published by {recommended_book['Publisher']} ({recommended_book['Year-Of-Publication']})</div>
                <div class="similarity-score">Similarity: {similarity_score:.1f}%</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
