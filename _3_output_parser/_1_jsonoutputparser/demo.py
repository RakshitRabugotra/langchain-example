import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login

# Output parser imports
from langchain_core.output_parsers import JsonOutputParser

required_env = ["HUGGINGFACEHUB_ACCESS_TOKEN"]

"""
What we're trying to do:
1. We will prompt to get the name, age and location of fictional character.
2. The LLM will generate a response, which is a string.
3. We want to parse the response using a JSON output parser.
"""

parser = JsonOutputParser()

# Create the template for the initial prompt
template = PromptTemplate(
    template="Give me the name, age and location of a fictional character.\n{format_instructions}",
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
        repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation"
    )
    return ChatHuggingFace(llm=llm, temperature=0.9, max_completion_tokens=100)


def example_without_chain():
    # Initialize the model
    print("Initializing the model...")
    model = init_model()
    print("Model initialized successfully.\n")

    # Prompt the model, and get the result
    prompt = template.format()
    result = model.invoke(prompt)
    print("Result from the model:\n", result.content)

    # Parse the result using the JSON output parser
    final_result = parser.parse(result.content)
    print("\nParsed Result:\n", final_result)


def example_with_chain():
    # Initialize the model
    print("Initializing the model...")
    model = init_model()
    print("Model initialized successfully.\n")

    # Create the chain
    chain = template | model | parser
    result = chain.invoke(
        {} # Since we have no input variables, we can pass an empty dict
    )
    print("Result from the model:\n", result)


# Set the main function
def main():
    # example_without_chain()
    example_with_chain()


exports = {
    'main': main,
    'env': required_env
}

if __name__ == "__main__":
    main()
