# NetflixRecommendation
**Tujuan:**   
Membangun model recommendation yang akan memberikan rekomendasi movie/tv show yang mirip dengan movie/tv show yang dimasukkan user.  
**Metodologi:**  
Projek ini menggunakan content-based recommender system yang memberikan rekomendasi berdasarkan kemiripan movie/tv show yang dimasukkan.   

**Sumber Dataset:**   
https://www.kaggle.com/datasets/anandshaw2001/netflix-movies-and-tv-shows   

**Fitur-fitur Dataset:**  
| Column | Deskripsi |
| --- | --- |
| show_id | Unique identifier untuk setiap show (s1, s2) |
| type | Menspesifikasikan jika show adalah movie atau tv show |
| title | Nama show | 
| director | Director show |
| cast | Aktor utama |
| country | Negara di mana show tersebut dibuat |
| date_added | Kapan show tersebut ditambahkan ke Netflix |
| release_year | Tahun terbit |
| rating | Content rating |
| duration | Durasi dalam menit untuk movie, dan dalam seasons untuk tv show |
| listed_in | Genre atau kategori | 
| description | Ringkasan show |  

**Proses Analisis dan Pemodelan:**
1.	Exploratory Data Analysis (EDA)
EDA dimulai dengan melihat dan memastikan tipe data setiap kolum, lalu mendeteksi dan memperbaiki kesalahan input. 
2.	Pembersihan dan Preprocessing Data:
Kolum ‘date_added’ awalnya memiliki tipe data object yang tidak dapat diolah secara langsung. Maka, kolum ‘date_added’ diubah menjadi recency score, yaitu total waktu sejak show tersebut ditambahkan ke Netflix. Selain itu, kolum ‘duration’ dipisah menjadi ‘duration_movie’ dan ‘duration_season’ untuk masing-masing move dan tv show. Terkahir, dilakukan identifikasi dan penanganan missing value.
3.	Persiapan Dataset
Dataset dibagi menjadi tiga berdasarkan jenis datanya, yaitu data TF-IDF, data kategoris, dan data numerik. Pada data TF-IDF, beberapa variabel penting seperti ‘title’ dan ‘listed_in’ diberikan weight yang lebih besar. Setelah itu, dilakukan vectorization. Data kategoris mengalami encoding menggunakan one hot encoder. Data numerik mengalami normalisasi. Terakhir, seluruh data digabungkan menjadi suatu sparse matrix. 
4.	Pemodelan
Kemiripan tiap show dengan show lain dihitung menggunakan cosine similarity dan disimpan ke suatu variabel. Setelah itu, dibuat recommendation function yang menerima judul show dan memberikan 5 rekomendasi. 
5.	Evaluasi Model
Evaluasi model dilakukan dengan melakukan testing dengan 3 show berbeda. Setelah itu, hasil rekomendasi dibandingkan dengan show input. 
6.	Deployment
Sebelum melakukan deployment pada streamlit, data yang menyimpan cosine similarity dan index dataset di-export menjadi file pkl. Setelah itu, file untuk tampilan streamlit dibuat dan diluncurkan lewat github.

**Hasil dan Kesimpulan:**
Berikut adalah contoh tampilan streamlit saat baru dibuka:
<img width="2559" height="933" alt="Screenshot 2025-09-11 104902" src="https://github.com/user-attachments/assets/57ac8b55-39e4-4742-b8e7-0b2d4571e574" />
Berikut adalah contoh hasil recommender system:
<img width="2539" height="1096" alt="Screenshot 2025-09-11 104933" src="https://github.com/user-attachments/assets/358532aa-9bfa-4db7-a01f-e490c0e03c8c" />
<img width="2547" height="843" alt="Screenshot 2025-09-11 104953" src="https://github.com/user-attachments/assets/cf8d34b4-ae2f-496b-bd3a-7d82dcdf800d" />
<img width="2553" height="850" alt="Screenshot 2025-09-11 105020" src="https://github.com/user-attachments/assets/37aa5ab1-5e91-4c8b-a026-1357fd9a7436" />
Secara umum, model yang dihasilkan dapat memberikan rekomendasi yang cukup mirip. Namun, model kadang terkecoh oleh kemiripan judul dan mengabaikan kemiripan lainnya.  

**Teknologi dan Library yang Digunakan:**  
•	pandas  
•	numpy  
•	scipy  
•	datetime  
•	matplotlib.pyploy  
•	scikit-learn   
•	streamlit  
