from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
import ast


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiples two floating point numbers
    """
    return a * b


def main():
    # Let's work with Gemini
    llm = init_chat_model(model="openai:gpt-4.1-mini")
    # Bind tools to the model
    llm_with_tools = llm.bind_tools([multiply])

    # Create a chat history for the llm
    chat_history = [
        
    ]

    while True:
        query = input("[HUMAN]: ")
        
        if query.lower() == "exit":
            break
            
        # Append to chat history
        chat_history.append(HumanMessage(query))
        # Call the model
        response: AIMessage = llm_with_tools.invoke(chat_history)
        chat_history.append(response)

        # If the response is a tool call, then say processing...
        if not response.additional_kwargs.get('tool_calls'):
            print(f"[AI]: {response.content}")
            continue
        
        print(f"[AI]: Processing...")
        # Process all the tool calls
        for tool_call in response.additional_kwargs['tool_calls']:
            tool_name = tool_call['function']['name']
            tool_args = ast.literal_eval(
                tool_call['function']['arguments']
            ) 
            print(f"\n[DEBUG] Invoking tool call: {tool_name}...")
            print(f"[DEBUG] Tool arguments: {tool_args}\n")
            # Invoke the tool
            tool_result = globals()[tool_name].invoke(tool_args)

            tool_response = ToolMessage(str(tool_result), name=tool_name, tool_call_id=tool_call['id'])
            # Append the tool call to the history
            chat_history.append(tool_response)
            
        # Call the model again
        response = llm_with_tools.invoke(chat_history)
        print(f"[AI]: {response.content}")
        # Append the response to the history
        chat_history.append(response)
                
                
exports = {
    "main": main,
    'env': ['OPENAI_API_KEY']
}

if __name__ == '__main__':
    main()


