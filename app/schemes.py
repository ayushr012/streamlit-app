import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from schemes_data import data

def suggest_similar_schemes(input_text, tfidf_matrix, input_tfidf, n=3):
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    similar_scheme_indices = similarities.argsort()[0][-n:][::-1]
    similar_scheme_names = [list(data.keys())[i] for i in similar_scheme_indices]
    return similar_scheme_names

def get_similar_schemes(input_text, n=3):
    tfidf_matrix, input_tfidf = extract_info(input_text)
    similar_schemes = suggest_similar_schemes(input_text, tfidf_matrix, input_tfidf, n)
    return similar_schemes

def extract_info(input_text):
    scheme_texts = []
    for data in data.values():
        text = ' '.join(data.values())
        scheme_texts.append(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(scheme_texts)
    input_tfidf = vectorizer.transform([input_text])
    return tfidf_matrix, input_tfidf

def main():
    st.title("Scheme Recommendation System")
    user_input = st.text_input("Enter your query:")
    
    if st.button("Get Similar Schemes"):
        if user_input:
            similar_schemes = get_similar_schemes(user_input)
            st.subheader("Similar Schemes:")
            for scheme in similar_schemes:
                st.write("- ", scheme)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()







