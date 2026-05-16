# !!!Note that index is already created and contains data on Pinecone!!!
# Import the relevant Python libraries
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec 

load_dotenv()

def create_embeddings(texts):
    """
    Creates embeddings out of provided texts

    Args:
        texts (List[str] or str): texts to be embedded

    Returns:
        List[List[float]]
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    response_dict = response.model_dump()
    return [data["embedding"] for data in response_dict["data"]]

def retrieve_closest(query):
    """
    Retrieves the closest claim to the query by meaning

    Args:
        query (List[float]): embedded query

    Returns:
        response to the query
    """
    closest = insurance_index.query(
        vector=query,
        top_k=1,
        include_metadata=True
    )

    return closest

# Load needed data
df = pd.read_csv("insurance_claims_top_100.csv")

# Connect to OpenAI and Pinecone via api
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Preparing data for Pinecone
ids = df["ClaimNumber"].tolist()
descriptions = df["ClaimDescription"].tolist()
metadatas = [{"age": row["Age"], "gender": row["Gender"]} for _, row in df.iterrows()]

# Embedding descriptions
embeds = create_embeddings(descriptions)

# Create index in Pinecone and connect to it
index_name = "insurance-index"
if not pc.has_index(name=index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
insurance_index = pc.Index(index_name) 

# Ingesting all data into the index
insurance_index.upsert(vectors=list(zip(ids, embeds, metadatas)))

query1 = "Car accident with rear-end collision"
query1_embedded = create_embeddings(query1)[0]

query2 = "Worker developed carpal tunnel syndrome from repetitive typing"
query2_embedded = create_embeddings(query2)[0]

# Finding the most semantically similar claims to queries
closest_claim = retrieve_closest(query1_embedded)
closest_claim_id = closest_claim["matches"][0]["id"]
closest_claim_description = df[df["ClaimNumber"] == closest_claim_id]["ClaimDescription"].values[0]

id = retrieve_closest(query2_embedded)["matches"][0]["id"]
closest_claim_description_carpal_tunnel = df[df["ClaimNumber"] == id]["ClaimDescription"].values[0]