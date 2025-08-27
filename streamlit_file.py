import streamlit as st
import joblib
import pandas as pd
import gdown
import os

# https://drive.google.com/file/d/13U-Iw-8tvBbddFgp-CXu5UJyfv-opXme/view?usp=sharing

@st.cache_resource
def load_data_from_drive():
    file_id = "13U-Iw-8tvBbddFgp-CXu5UJyfv-opXme"
    output_path = "recommender_model.pkl"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        return joblib.load(f)

recommender_data = load_data_from_drive()

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
        st.subheader(idx+1, ". ", row['title'])
        st.text("Genre: ", row['listed_in'])
        st.text("Rating: ", row['rating'])
        st.text("Description: ", row['description'])

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
