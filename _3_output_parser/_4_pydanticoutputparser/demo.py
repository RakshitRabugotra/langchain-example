import os
from pydantic import BaseModel, Field

# LangChain imports
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

required_env = ["HUGGINGFACEHUB_ACCESS_TOKEN", "HUGGINGFACE_MODEL"]

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=0, description="Age of the person")
    city: str = Field(description="City where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name, age and city of a fictional {place} person.\n{format_instructions}",
    input_variables=["place"], 
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def init_model():
    """
    Initialize the HuggingFace model for text generation.
    """
    # Login using the env variables to huggingface
    login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

    # Create a huggingface endpoint for the model
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("HUGGINGFACE_MODEL"), task="text-generation"
    )
    return ChatHuggingFace(llm=llm, temperature=0.9, max_completion_tokens=100)

def main():
    # Initialize the model
    print("Initializing the model...")
    model = init_model()
    print("Model initialized successfully.\n")

    # Create the chain
    chain = template | model | parser

    # Invoke the chain with a specific place
    result = chain.invoke({"place": "Indian"})
    
    # If you want to access the attributes of the Person object
    if result and isinstance(result, Person):
        print(f"Name: {result.name}, Age: {result.age}, City: {result.city}")

exports = {
    'main': main,
    'env': required_env
}

if __name__ == "__main__":
    main()