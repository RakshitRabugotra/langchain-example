import os

# Text splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# To use an llm
from langchain.chat_models import init_chat_model

# To generate embeddings and store in vector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# import the web base loader of document
from langchain_community.document_loaders import SeleniumURLLoader

# To use the qa chain from langchain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Utility imports
from common_utility import save_if_yes

# The knowledge will be extracted from these urls
urls = [
    "https://uttarakhandtourism.gov.in/public/blogs/best-adventures-to-try-in-uttarakhand",
    "https://uttarakhandtourism.gov.in/public/blogs/kumbh-mela",
    "https://uttarakhandtourism.gov.in/public/blogs/beatles-magical-tour-of-india",
]

required_env = ["GOOGLE_API_KEY", "OPENAI_API_KEY"]

# Our document loader
document_loader = SeleniumURLLoader(
    urls=urls,
    browser="chrome",
    # binary_location='./res/bin/chrome/chromedriver.exe'
)

# TODO: Change this path to your database
# Configure the vectorstore
DB_PATH = "./res/db"

if not os.path.isdir(DB_PATH):
    os.makedirs(DB_PATH)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=140)


# Creates a chrome vector store
def init_vectorstore() -> Chroma:
    return Chroma(
        embedding_function=OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
        ),
        persist_directory=os.path.join(DB_PATH, "chroma_db"),
        collection_name="tourism",
    )


def main():

    # Initialize the vector store
    vectorstore = init_vectorstore()

    # Get all the document id's stored in the
    document_ids = vectorstore.get()["ids"]

    is_store_empty = len(document_ids) == 0
    # If the vectorstore is empty, then load the documents and fill it
    if is_store_empty:
        print("Vector store is empty, fetching documents...")

        # Load the documents and print the metadata
        documents = document_loader.load()

        print(f"Fetched: {len(documents)}\nSplitting into chunks...")
        # Now we will split the text based on paragraphs
        documents = text_splitter.split_documents(documents)
        print()

    print("Documents loaded!")
    # Get the model name for LLM
    llm_model_name = os.getenv("GOOGLE_GENERATIVE_MODEL", "gemini-2.5-flash")

    # Store the splits
    if is_store_empty:
        vectorstore.add_documents(documents)

    # Create the gemini model
    # llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0.5)
    llm = init_chat_model(
        model=llm_model_name, model_provider="google_genai", temperature=0.5
    )

    # Get the qa prompt
    retrieval_qa_chat_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "Answer any use questions based solely on the context below:\n<context>\n{context}\n</context>",
            ),
            ("human", "{input}"),
        ]
    )

    # Create a chain for parsing documents
    combine_docs_chain = create_stuff_documents_chain(
        llm, prompt=retrieval_qa_chat_prompt
    )

    # The RAG chain, the resultant would be stored in 'answer' key
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    # Now we can get the user's query
    query = input("Enter your query: ")
    output = rag_chain.invoke({"input": query})

    # Print only the output
    print(f"[AI]: {output['answer']}")

    # Ask the user if they want to save the file
    save_if_yes(
        content=output["answer"],
        # TODO: Modify this save directory path
        save_directory="./out",
        # TODO: Modify the extension if you want to
        ext=".md",
    )


# Ignore this, doesn't affect this module when ran individually
exports = {"main": main, "env": required_env}

if __name__ == "__main__":
    main()
