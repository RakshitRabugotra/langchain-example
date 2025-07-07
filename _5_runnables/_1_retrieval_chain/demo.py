import os

# LangChain community imports
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OpenAI model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

required_env = ["OPENAI_API_KEY", "OPENAI_MODEL"]

"""
What we want to do is:
1. Load the text document into memory,
2. 
"""

# Load the document
path_to_doc = "./res/doc.txt"
loader = TextLoader(path_to_doc, encoding='utf-8')
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

def main():
    # Convert text into embeddings & store in FAISS
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(model="text-embedding-3-large"))

    # Create a retriever (fetches relevant documents)
    retriever = vectorstore.as_retriever()

    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    # Create a RetrievalQAChain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Manually Retrieve relevant documents
    query = "What are the key takeaways from the document?"

    print("Generating response...")
    answer = qa_chain.invoke(query)    
    print("LLM Response\n:", answer)


# The files exports
exports = {"main": main, "env": required_env}

if __name__ == "__main__":
    main()
