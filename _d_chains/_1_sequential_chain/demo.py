import os

# Pydantic imports
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Hugging face model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login

required_env = ["HUGGINGFACEHUB_ACCESS_TOKEN", "HUGGINGFACE_MODEL"]


# We will create a pydantic class to structure the output
class PoemResponse(BaseModel):
    poem: str = Field(description="The generated poem for the given topic")
    poem_theme: str = Field(description="The theme of the poem captured in a small sentence")
    poetic_device_highlights: str = Field(
        description="The highlights in the poem where the poetic device is used"
    )
    word_count: int = Field(gt=0, description="The number of words used in the poem")

# Create a parser for this response
parser = PydanticOutputParser(pydantic_object=PoemResponse)

# Create a prompt to generate a poem on given topic
template = PromptTemplate(
    template="""
  You are an expert creative writer who has written 100s of stories and poems. 
  Write a poem on {topic}. Make sure to use {poetic_devices}\n{format_instructions}
  """,
    input_variables=["topic", "poetic_devices"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)


def init_model():
    # Initialize the login,
    login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

    # Create a huggingface endpoint for the model
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("HUGGINGFACE_MODEL"),
        task="text-generation",
        model_kwargs=dict(max_completion_tokens=200),
    )
    return ChatHuggingFace(llm=llm, temperature=0.9)


def main():
    # We will create a simple sequential chain
    model = init_model()

    # Create the chain
    chain = template | model | parser

    # Ask the user, which topic and poetic devices to use
    topic = input("Enter the topic for poem: ")
    poetic_devices = input("Enter poetic devices separated by commas: ")

    # Get the final result
    print("Generating result...")

    result = chain.invoke(dict(topic=topic, poetic_devices=poetic_devices))
    # Now th response should be an object of PoemResponse
    if result and isinstance(result, PoemResponse):
        # We will print the poem and the metadata
        print("Here is your requested poem:\n", result.poem)
        print("\nThe theme of the poem is: ", result.poem_theme)
        print("\nThe poetic device highlights: ", result.poetic_device_highlights)
        print("\nThe word count is: ", result.word_count)


# The files exports
exports = {"main": main, "env": required_env}

if __name__ == "__main__":
    main()
