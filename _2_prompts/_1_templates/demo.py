from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

def prompt_template():
    """
    This function demonstrates how to create a prompt template using LangChain.
    It uses the PromptTemplate class to define a template with a placeholder for user input.
    """
    template = "What is the capital of {country}?"
    prompt = PromptTemplate(input_variables=["country"], template=template)
    
    # Example usage
    country = "France"
    formatted_prompt = prompt.invoke(country)
    
    print(f"Formatted Prompt: {formatted_prompt}")

def chat_prompt_template():
    """
    This function demonstrates how to create a chat prompt template using LangChain.
    It uses the PromptTemplate class to define a chat template with a placeholder for user input.
    """
    chat_prompt = ChatPromptTemplate([
        ("system", "You are a helpful assistant."),
        ("user", "{user_input}"),
    ], input_variables=["user_input"])
    
    # Example usage
    user_input = "What is the capital of France?"
    formatted_chat_prompt = chat_prompt.invoke(user_input)
    
    print(f"Formatted Chat Prompt: {formatted_chat_prompt}")



def main():
    print("Running prompt template examples...")
    
    # Run the prompt template example
    prompt_template()
    
    print("\nRunning chat prompt template example...")
    
    # Run the chat prompt template example
    chat_prompt_template()
    
    print("\nPrompt templates executed successfully.")
  
exports = {
    "main": main,
    "env": None
}

if __name__ == "__main__":
    main()