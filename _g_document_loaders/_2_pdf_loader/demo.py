import os

# Langchain import
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Runnables
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Strong output parsers
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel

# Gemini Model
from langchain_google_genai import ChatGoogleGenerativeAI


# We can give the user option to load specific pdf to generate the system prompt
def get_file_path(base_path="./res/pdfs"):
    """
    Gets any specific pdf file path from resources folder
    """
    pdfs = [file for file in os.listdir(base_path) if file.endswith(".pdf")]

    print("Choose a pdf from the following: ")
    for i, file in enumerate(pdfs):
        print(f"[{i}] ", file)

    choice = -1
    while choice < 0:
        try:
            choice = int(input("\nEnter your choice: "))
        except Exception as e:
            print("Invalid choice try again!")

    # With a valid choice, we can load that pdf
    file_path = os.path.join(base_path, pdfs[choice])
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error finding the file `{file_path}`") from None

    return file_path


"""
For information, the `Document` object of LangChain has a schema like this:
Document(
  page_content="The actual text content",
  metadata={"source": "filename.pdf", ...}
)
"""
required_env = ["GOOGLE_API_KEY", "GOOGLE_GENERATIVE_MODEL"]


# To get the system prompt
class SystemPromptResponse(BaseModel):
    system_prompt: str = Field(
        description="The system prompt generated from the request"
    )


# Create a parser
system_prompt_parser = PydanticOutputParser(pydantic_object=SystemPromptResponse)


def document_loader(inputs=None):
    loader = PyPDFLoader(
        file_path=get_file_path(),
    )

    # Gets the array of documents
    documents = loader.load()
    return {"documents": documents}


def print_metadata(inputs: dict):
    documents = inputs["documents"]
    # We can get the number of documents generated
    count = len(documents)
    print("Number of documents: ", count)

    # Print the first 100 characters of the content
    print("Content: ", documents[0].page_content[:100])
    # Print the doc that we got
    print("Metadata: ", documents[0].metadata)
    return inputs


def format_documents(inputs: dict):
    return {"content": "\n".join([doc.page_content for doc in inputs["documents"]])}


def save_if_positive(inputs: SystemPromptResponse):
    choice = input("Do you want to save the result to file? (y/N): ")
    if choice == "N":
        return inputs
    # Else, ask the filename, and save it
    filename = input("Enter filename (without .txt): ")
    with open(os.path.join("./out/system-prompts", filename), mode="w+") as file:
        file.write(inputs.system_prompt)
    return inputs


def main():

    # Create Gemini model to use :)
    gemini = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_GENERATIVE_MODEL"), temperature=0.9)

    # Create an output parser
    parser = StrOutputParser()

    # Creating a prompt template
    prompt = PromptTemplate(
        template="From the given description of the character create a strong SYSTEM PROMPT to feed to a CHAT MODEL, which PLAYS THE CHARACTER SOULFULLY. Description: {content}\n{format_instructions}",
        input_variables=["content"],
        partial_variables={
            "format_instructions": system_prompt_parser.get_format_instructions()
        },
    )

    # Create the pipeline for system prompt
    system_prompt_chain = (
        RunnableLambda(document_loader)
        | RunnableLambda(print_metadata)
        | RunnableLambda(format_documents)
        | prompt
        | gemini
        | system_prompt_parser
        | RunnableLambda(save_if_positive)
    )

    print("\nGenerating system prompt")
    pydantic_object: SystemPromptResponse = system_prompt_chain.invoke({})

    # Feed the system prompt to the chat
    sys_prompt = (
        pydantic_object.system_prompt + "\nONLY REPLY IN SHORT SENTENCES 3 to 5 AT MAX"
    )
    # Print the system prompt
    print("System Prompt Generated:\n" + sys_prompt + "\n")

    # Creating a chat template from the system prompt
    chat_prompt = ChatPromptTemplate(
        [("system", "{system_prompt}"), ("user", "{user_input}")],
        input_variables=["user_input"],
        partial_variables={"system_prompt": sys_prompt},
    )

    # Get the result after invoking
    print("\nCreating chat...")
    chat_chain = chat_prompt | gemini | parser

    user_input = input("Let's chat: ")
    result = chat_chain.invoke({"user_input": user_input})
    print("AI: ", result)


# The exports from this demo
exports = {"main": main, "env": required_env}

if __name__ == "__main__":
    main()
