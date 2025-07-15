import os
import time

# Document loaders
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Custom import
from loaders import MenuLoader


# The base path where all the pdfs are stored
BASE_PATH = "./res/books"

# Just so we don't get a directory not found error
if not os.path.isdir(BASE_PATH):
    os.makedirs(BASE_PATH)


def load_documents(is_lazy):
    # Create a counter
    start = time.time()
    # Create an instance of the document loader
    document_loader = DirectoryLoader(
        path=BASE_PATH,
        glob="**/*.pdf",  # Search all the subdirectories recursively
        loader_cls=PyPDFLoader,
    )

    # Print the Documents metadata that we got
    documents = document_loader.lazy_load() if is_lazy else document_loader.load()
    end = time.time()

    # Print the time it took to load the files
    print(f"Time took to load: {end-start:.9f}s")


def main():
    # Create a demo to optionally run both
    options = [
        # We just want the choice, so no need to call the functions here
        ("Load", False),
        ("Lazy Load", True),
    ]

    # Run the options
    choice = MenuLoader.run_options(
        options, title="Choose the load function", return_type="choice"
    )
    # Get the flag
    _, flag = options[choice]
    # Run the selected option
    load_documents(is_lazy=flag)


# To run the program globally, this is a standard way I created
exports = {"main": main, "env": None}

if __name__ == "__main__":
    main()
