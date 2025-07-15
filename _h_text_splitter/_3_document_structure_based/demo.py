
# Import text splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
# Document load
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Let's set our example to work with python files
BASE_CODE_PATH = "./res/source-codes"
EXT = "py"

# Create an instance of the splitter
# Now we're using a classmethod to form the splitter from the language
splitter = RecursiveCharacterTextSplitter.from_language(
  language=Language.PYTHON,
  chunk_size=500,
  chunk_overlap=0,
)

# Create a document loader, which loads all the given source code files
document_loader = DirectoryLoader(
  path=BASE_CODE_PATH,
  glob="**/*." + EXT,
  loader_cls=TextLoader
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