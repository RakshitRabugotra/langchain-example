import os

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import login

required_env = ["HUGGINGFACEHUB_ACCESS_TOKEN", "HUGGINGFACE_MODEL"]

"""
What we're trying to do:
1. We will prompt to get 5 benefits of meditation.
2. The LLM will generate a response, which is a string.
3. We want to parse the response using a structured output parser.
4. The output will be a dictionary with the benefits as keys and their descriptions as values.
"""

schema = [
    ResponseSchema(name='benefit-1', description='Benefit 1 about the topic', type='string'),
    ResponseSchema(name='benefit-2', description='Benefit 1 about the topic', type='string'),
    ResponseSchema(name='benefit-3', description='Benefit 1 about the topic', type='string'),
    ResponseSchema(name='benefit-4', description='Benefit 1 about the topic', type='string'),
    ResponseSchema(name='benefit-5', description='Benefit 1 about the topic', type='string')
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Create the template for the initial prompt
template = PromptTemplate(
    template="Give me 5 benefits of meditation.\n{format_instructions}",
    input_variables=[],  # It is going to be a static prompt, so no input variables
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

    final_result = chain.invoke(
        {}  # Since we have no input variables, we can pass an empty dict
    )
    print("\nParsed Result:\n", final_result)

exports = {
    'main': main,
    'env': required_env
}

if __name__ == "__main__":
    main()