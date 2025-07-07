import os
from langchain_community.document_loaders import TextLoader

# Relative import for this module (as we'll be running this from `main.py`)
from .file_encoding import get_file_encoding

file_path = "./res/subtitles_english.txt"
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"Error finding the file `{file_path}`") from None


"""
For information, the `Document` object of LangChain has a schema like this:
Document(
  page_content="The actual text content",
  metadata={"source": "filename.pdf", ...}
)
"""


def main():
    loader = TextLoader(
        file_path=file_path,
        # This is the encoding of this file, may differ for you
        encoding=get_file_encoding(file_path).lower(),
    )

    # Gets the array of documents
    documents = loader.load()

    # We can get the number of documents generated
    count = len(documents)
    print("Number of documents: ", count)


# The exports from this demo
exports = {"main": main, "env": None}

if __name__ == "__main__":
    main()
