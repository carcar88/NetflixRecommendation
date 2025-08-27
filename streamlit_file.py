import streamlit as st
import joblib
import pandas as pd

recommender_data = joblib.load('data_similarity.pkl')

cosine_similarities = recommender_data['cosine_similarities']
indices = recommender_data['indices']
df = pd.read_csv('netflix_preprocessed.csv')

# buat recommender function yang secara langsung menunjukkan rekomendasi yang diberikan dalam format streamlit
def content_recommender_streamlit(name):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[2:7]
    netflix_indices = [i[0] for i in sim_scores]
    displayed_column = ['title', 'listed_in', 'description', 'rating']
    recommendations = df.iloc[netflix_indices][displayed_column]

    st.subheader(f"🎥 Recommended Titles Similar to **{name}**")
    for idx, row in recommendations.iterrows():
        st.markdown(f"""
        ### 🎬 {row['title']}
        Genre: {row['listed_in']}
        Rating: {row['rating']}
        Description: {row['description']}
        ---
        """)

def main():
    st.title('Netflix Recommendation')

    # Input judul yang ingin dicari
    title = st.text_input("Find movies/tv shows similar to", placeholder="ex: Final Destination")

    if st.button("Find Recommendations"):
        # preprocess title yang dimasukkan dan deteksi jika kosong
        if title == '' or not title:
            st.warning('Please input a movie/tv show title!')
            return
        title = title.title()
        content_recommender_streamlit(title)

if __name__ == '__main__':
    main()
