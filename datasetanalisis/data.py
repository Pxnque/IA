import pandas as pd
import os 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

file_path = 'datasetTexto.csv'

if os.path.exists(file_path):
    
    sample_df = pd.read_csv(file_path, on_bad_lines='skip')
    print(sample_df.head())
    print(f"\nColumn names:")
    print(sample_df.columns.tolist())

    print("\nFirst 5 rows:")
    print(sample_df.head())

    print("\nData types:")
    print(sample_df.dtypes)
    print(sample_df)
else:
    print(f"Error: The file '{file_path}' was not found.")

for i in sample_df:
    results.append(i[5])

vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words='spanish',
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(results)
similarity_matrix = cosine_similarity(X)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)
print(clusters)
