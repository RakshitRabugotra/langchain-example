
# Document loaders
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# The base path where all the pdfs are stored
PDF_BASE_PATH = "./res/pdfs"

def main():
  # Create an instance of the document loader
  document_loader = DirectoryLoader(
    path=PDF_BASE_PATH,
    glob="**/*.pdf", # Search all the subdirectories recursively
    loader_cls=PyPDFLoader
  )
  
  # Print the Documents metadata that we got
  documents = document_loader.load()

  print("Here is the loaded document: ")
  for i, doc in enumerate(documents):
    print(f"{i} => {doc.metadata} | Word Count: {len(doc.page_content)}\n")

# To run the program globally, this is a standard way I created
exports = {
  'main': main,
  'env': None
}

if __name__ == '__main__':
  main()