import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from schemes_data import schemes_data

def suggest_similar_schemes(input_text, tfidf_matrix, input_tfidf, n=3):
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    similar_scheme_indices = similarities.argsort()[0][-n:][::-1]
    similar_scheme_names = [list(schemes_data.keys())[i] for i in similar_scheme_indices]
    return similar_scheme_names

def get_similar_schemes(input_text, n=3):
    tfidf_matrix, input_tfidf = extract_info(input_text)
    similar_schemes = suggest_similar_schemes(input_text, tfidf_matrix, input_tfidf, n)
    return similar_schemes

def extract_info(input_text):
    scheme_texts = []
    for data in schemes_data.values():
        text = ' '.join(data.values())
        scheme_texts.append(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(scheme_texts)
    input_tfidf = vectorizer.transform([input_text])
    return tfidf_matrix, input_tfidf


# Set page configuration
st.set_page_config(page_title="🌾 Agriculture Scheme Recommendation", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #edfaf9;
    }
    .title {
        font-size: 50px;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 24px;
        color: #161716;
        text-align: center;
        margin-bottom: 20px;
    }
    .query-box {
        width: 80%;
        margin: 0 auto;
        text-align: center;
    }
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .recommend-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .scheme-item {
        font-size: 18px;
        font-weight: bold;
        color: #228B22;
    }
    </style>
""", unsafe_allow_html=True)

# Main function
def main():
    # Title and Subheader
    st.markdown('<p class="title">🌾 Agriculture Scheme Recommendation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Get the Best Government Schemes for Your Agricultural Needs</p>', unsafe_allow_html=True)
    
    # Creating layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        
        user_input = st.text_input("🔍 Enter your agriculture query:", placeholder="e.g., Loans for farmers")
        st.markdown('</div>', unsafe_allow_html=True)

        # Stylish Button
        if st.button("🌱 Get Similar Schemes", key="recommend_button"):
            if user_input:
                similar_schemes = get_similar_schemes(user_input)
                st.markdown('<p class="subheader">✅ Recommended Schemes:</p>', unsafe_allow_html=True)
                for scheme in similar_schemes:
                    st.success(f"🌟 {scheme}")
            else:
                st.warning("⚠️ Please enter a query.")

if __name__ == "__main__":
    main()








