# This project is a LangChain example for document similarity search using embedding models.
import numpy as np
import os

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
# Hugging Face Hub imports
from huggingface_hub import login

# Ensure required env
required_env = ["OPENAI_API_KEY", "HUGGINGFACEHUB_ACCESS_TOKEN"]

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]


def main():

  print("Logging into Hugging Face Hub...")
  # Have to call login function and provide your Hugging Face token
  login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

  print("Creating HuggingFaceEmbeddings instance...")
  embedding = HuggingFaceEmbeddings(
      model="sentence-transformers/all-MiniLM-L6-v2",
      model_kwargs={"device": "cpu"},
      encode_kwargs={"normalize_embeddings": True}
  )
  print("Embedding documents...")
  document_embeddings = embedding.embed_documents(document)

  while True:
    query = input("\n\nEnter your query: ")
    if not query:
        print("Exiting the program.")
        break

    print("Embedding your query...")
    query_embedding = embedding.embed_query(query)

    print("Calculating cosine similarity...")
    scores = cosine_similarity(
        [query_embedding], document_embeddings
    )[0]

    print("\n\nThe highest similarity score is:")
    max_score = np.max(scores)
    max_scoring_document = document[np.argmax(scores)]
    print(f"Document: \"{max_scoring_document}\"\nScore: {max_score:.4f}")

exports = {
    'main': main,
    'env': required_env
}

if __name__ == "__main__":
  main()