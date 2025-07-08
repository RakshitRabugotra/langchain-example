import os

# Langchain import
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Gemini Model
from langchain_google_genai import ChatGoogleGenerativeAI

# Huggingface model
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from huggingface_hub import login

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
required_env = ["HUGGINGFACEHUB_ACCESS_TOKEN", "HUGGINGFACE_MODEL"]


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

    # Print the first 100 characters of the content
    print("Content: ", documents[0].page_content[:100])
    # Print the doc that we got
    print("Metadata: ", documents[0].metadata)

    # Create the model
    # llm = HuggingFaceEndpoint(
    #     repo_id=os.getenv("HUGGINGFACE_MODEL", "deepseek-ai/DeepSeek-R1-0528"),
    #     task="text-generation",
    # )
    # huggingface = ChatHuggingFace(llm=llm, temperature=0.9)

    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)

    # Create an output parser
    parser = StrOutputParser()

    # Creating a prompt template
    prompt = PromptTemplate(
        template="Write a summary about the characteristics, traits, tendency, working of the character {character} from its dialogues (if the character is not in the script, state so)\nScript:\n{dialogues}",
        input_variables=["character", "dialogues"],
    )

    # Create the pipeline for summary
    # summary_chain = prompt | huggingface | parser
    summary_chain = prompt | gemini | parser

    # Get the character from the user
    character = input("Enter the character to summarize: ")

    # Get the result after invoking
    print("\nGenerating summary...")
    result = summary_chain.invoke(
        {"character": character, "dialogues": documents[0].page_content}
    )

    # If the result is very long, then output in file
    if len(result) <= 100:
        print("Result: ", result)
        exit(0)

    # If not so, then write to a file
    filename = input("Result is too long, enter a filename (without .txt): ")
    with open(f"./out/{filename}.md", mode="w+") as file:
        file.write(result)
    print("\nResult saved to the file: " + f"out/{filename}.md")


# The exports from this demo
exports = {"main": main, "env": required_env}

if __name__ == "__main__":
    main()
