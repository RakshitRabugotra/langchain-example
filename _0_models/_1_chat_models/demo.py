import os

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)

# Ensure the required_env
required_env = [
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "GOOGLE_API_KEY",
    "HUGGINGFACEHUB_ACCESS_TOKEN"
]

# Custom utilities
from loaders import MenuLoader

def chat_openai():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0.9,max_completion_tokens=100)
    prompt = input("Prompt: ")  # "What is the capital of France?"
    response = llm.invoke(prompt)
    print(response.content)


def chat_anthropic():
    llm = ChatAnthropic(model="claude-2", temperature=0.9)
    prompt = input("Prompt: ")  # "What is the capital of France?"
    response = llm.invoke(prompt)
    print(response.content)


def chat_google():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)
    prompt = input("Prompt: ")  # "What is the capital of France?"
    response = llm.invoke(prompt)
    print(response.content)


def chat_huggingface_api():
    from huggingface_hub import login

    # Have to call login function and provide your Hugging Face token
    login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
    )
    llm = ChatHuggingFace(llm=llm)
    prompt = input("Prompt: ")  # "What is the capital of France?"
    response = llm.invoke(prompt)
    print(response.content)


def chat_huggingface_local():
    from huggingface_hub import login

    # Have to call login function and provide your Hugging Face token
    login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

    # Set the cache directory for Hugging Face models
    os.environ["HF_HOME"] = os.path.join(os.getcwd(), "cache/huggingface")

    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={"temperature": 0.9, "max_new_tokens": 100},
    )

    model = ChatHuggingFace(llm=llm, task="text-generation")
    prompt = input("Prompt: ")  # "What is the capital of France?"
    response = model.invoke(prompt)
    print(response.content)


def main():

    # Create a list of options from the global functions that start with "chat_"
    options = MenuLoader.create_options_from_globals(
        globals_dict=globals(),
        prefix="chat_", format_to_title=True
    )
    if not options:
        print("No chat models available. Please check your environment setup.")
        return

    # Initialize the MenuLoader with the options
    MenuLoader.run_options(options, title="Chat Model Demo")


exports = {
    'main': main,
    'env': required_env
}

if __name__ == "__main__":
    main()
