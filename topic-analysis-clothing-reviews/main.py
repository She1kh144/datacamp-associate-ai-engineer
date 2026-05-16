from scipy.spatial import distance
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load environment variables
load_dotenv()

# Load and filter the dataset
reviews = pd.read_csv("womens_clothing_e-commerce_reviews.csv")
filtered_reviews = reviews[reviews['Review Text'].notna()]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_embeddings(texts):
    """
    Creates embeddings out of provided texts

    Args:
        texts (List[str]): texts to be embedded

    Returns:
        List[List[float]]
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    response_dict = response.model_dump()
    return [data["embedding"] for data in response_dict["data"]]

def find_closest(query_vector, embeddings, n=1):
    """Returns the n closest items based on cosine distance"""
    distances = []
    for idx, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"dist": dist, "idx": idx})
    
    sorted_distances = sorted(distances, key=lambda x: x["dist"])
    return sorted_distances[:n]

# Embed every review text
filtered_reviews_texts = [data["Review Text"] for _, data in filtered_reviews.iterrows()]
embeddings = create_embeddings(filtered_reviews_texts)
    
# Reduce embeddings to 2D and visualize it
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(np.array(embeddings))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.show()

# Categorization
categories = [
    {"name": "quality", "description": "quality of a dress"},
    {"name": "fit", "description": "fit of a dress"},
    {"name": "style", "description": "style of a dress"},
    {"name": "comfort", "description": "comfort of a dress"},
]

# Embedding part
class_descriptions = [category["description"] for category in categories]
class_embeddings = create_embeddings(class_descriptions)

# Classifying first 10 reviews
for i in range(10):
    closest = find_closest(embeddings[i], class_embeddings)[0]
    category = categories[closest["idx"]]["name"]
    print(f"{i+1}: {category}")
    print(filtered_reviews_texts[i])

# Similarity search function
closest = find_closest(embeddings[0], embeddings, n=3)
most_similar_reviews = [filtered_reviews_texts[data["idx"]] for data in closest]
print(most_similar_reviews)