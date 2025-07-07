import os

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# Hugging Face Hub imports
from huggingface_hub import login

# Ensure required env
required_env = ["OPENAI_API_KEY", "HUGGINGFACEHUB_ACCESS_TOKEN"]

# Custom utilities
from loaders import MenuLoader

def __create_text_document(max_inputs: int = 3):
    """
    Create a sample text document for embedding.
    """
    
    document, inputs = [], max_inputs
    print(f"Please enter {max_inputs} text inputs for the document:")
    while inputs:  
        text = input(f"[input-{max_inputs-inputs+1}]: ")  # "What is the capital of France?"
        document.append(text)
        inputs -= 1
    
    return document

def embedding_openai_query():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
    text = input("Text: ")  # "What is the capital of France?"
    response = embeddings.embed_query(text)
    print(response)

def embedding_openai_document():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
    document = __create_text_document()
    response = embeddings.embed_documents(document)
    print(response)


def embedding_huggingface_local_query():
    # Have to call login function and provide your Hugging Face token
    login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    text = input("Text: ")  # "What is the capital of France?"
    response = embeddings.embed_query(text)
    print(response)

def embedding_huggingface_local_documents():

    # Have to call login function and provide your Hugging Face token
    login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    document = __create_text_document()
    response = embeddings.embed_documents(document)
    print(response)

def main():
    # Create a list of options from the global functions that start with "chat_"
    options = MenuLoader.create_options_from_globals(
        globals_dict=globals(),
        prefix="embedding_", format_to_title=True
    )
    if not options:
        print("No chat models available. Please check your environment setup.")
        return

    # Initialize the MenuLoader with the options
    MenuLoader.run_options(options, title="Embedding Model Demo")

exports = {
    'main': main,
    'env': required_env
}

if __name__ == "__main__":
    # Run the embedding demo
    main()