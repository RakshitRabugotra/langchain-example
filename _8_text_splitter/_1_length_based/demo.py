
# Import text splitters
from langchain.text_splitter import CharacterTextSplitter
# Document load
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

BASE_PDF_PATH = "./res/pdfs"

# Create an instance of the splitter
splitter = CharacterTextSplitter(
  chunk_size=100,
  chunk_overlap=0,
  separator=''
)

# Create a document loader
document_loader = DirectoryLoader(
  path=BASE_PDF_PATH,
  glob="**/*.pdf",
  loader_cls=PyPDFLoader
)

def main():

  # Load the documents
  documents = document_loader.load()

  # Split the text
  result = splitter.split_documents(documents)

  for i, chunk in enumerate(result):
    print(f"{i} => ", chunk.page_content)

# Create exports to run the program
exports = {
  'main': main,
  'env': None
}