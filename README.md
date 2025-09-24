# Netflix Recommendation Model
**Tujuan:**   
Membangun model _recommendation_ yang akan memberikan rekomendasi _movie_/_tv show_ yang mirip dengan movie/tv show yang dimasukkan _user_.  
**Metodologi:**  
Projek ini menggunakan _content-based recommender system_ yang memberikan rekomendasi berdasarkan kemiripan _movie_/_tv show_ yang dimasukkan.   

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
1.	_Exploratory Data Analysis_ (EDA)
EDA dimulai dengan melihat dan memastikan tipe data setiap kolum, lalu mendeteksi dan memperbaiki kesalahan _input_. 
2.	Pembersihan dan _Preprocessing_ Data:
Kolum ‘date_added’ awalnya memiliki tipe data _object_ yang tidak dapat diolah secara langsung. Maka, kolum ‘date_added’ diubah menjadi_ recency score_, yaitu total waktu sejak show tersebut ditambahkan ke Netflix. Selain itu, kolum ‘duration’ dipisah menjadi ‘duration_movie’ dan ‘duration_season’ untuk masing-masing _movie_ dan _tv show_. Terkahir, dilakukan identifikasi dan penanganan _missing value_.
3.	Persiapan _Dataset_
_Dataset_ dibagi menjadi tiga berdasarkan jenis datanya, yaitu data TF-IDF, data kategoris, dan data numerik. Pada data TF-IDF, beberapa variabel penting seperti ‘title’ dan ‘listed_in’ diberikan _weight_ yang lebih besar. Setelah itu, dilakukan _vectorization_. Data kategoris mengalami _encoding_ menggunakan _one hot encoder_. Data numerik mengalami normalisasi. Terakhir, seluruh data digabungkan menjadi suatu _sparse matrix_. 
4.	Pemodelan
Kemiripan tiap _show_ dengan _show_ lain dihitung menggunakan _cosine similarity_ dan disimpan ke suatu variabel. Setelah itu, dibuat _recommendation function_ yang menerima judul _show_ dan memberikan 5 rekomendasi. 
5.	Evaluasi Model
Evaluasi model dilakukan dengan melakukan _testing_ dengan 3 _show_ berbeda. Setelah itu, hasil rekomendasi dibandingkan dengan _show input_. 
6.	_Deployment_
Sebelum melakukan _deployment_ pada streamlit, data yang menyimpan _cosine similarity_ dan_ index dataset_ di-_export_ menjadi _file_ pkl. Setelah itu, _file_ untuk tampilan streamlit dibuat dan diluncurkan lewat github.

**Hasil dan Kesimpulan:**
Berikut adalah contoh tampilan streamlit saat baru dibuka:
<img width="2559" height="933" alt="Screenshot 2025-09-11 104902" src="https://github.com/user-attachments/assets/57ac8b55-39e4-4742-b8e7-0b2d4571e574" />
Berikut adalah contoh hasil_ recommender system_:
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
