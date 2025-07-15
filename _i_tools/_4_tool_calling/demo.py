from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiples two floating point numbers
    """
    return a * b


def repeat_calling_till_no_tool_calls(
    llm_with_tools: Any, chat_history: list[BaseMessage]
) -> AIMessage:
    """
    Repeats calling the model till there are no tool calls
    """
    # Assume if don't know the conversion_rate
    while True:
        response: AIMessage = llm_with_tools.invoke(chat_history)
        chat_history.append(response)

        # If the response is a tool call, then say processing...
        if not response.tool_calls:
            return response

        print(f"[DEBUG] Tool calls: {response.tool_calls}")
        # NOTE: tool_calls is a list of `tool_call` which is a dict
        # Process all the tool calls
        for tool_call in response.tool_calls:
            print(f"[DEBUG] Tool call: {tool_call}")
            # Invoke the tool
            tool_result = globals()[tool_call["name"]].invoke(tool_call)
            chat_history.append(tool_result)


def main():
    # Let's work with Gemini
    llm = init_chat_model(model="openai:gpt-4.1-mini")
    # Bind tools to the model
    llm_with_tools = llm.bind_tools([multiply])

    # Create a chat history for the llm
    chat_history = []

    while True:
        query = input("[HUMAN]: ")

        if query.lower() == "exit":
            break

        # Append to chat history
        chat_history.append(HumanMessage(query))

        # Call the model, repeatedly till there are no tool calls
        response = repeat_calling_till_no_tool_calls(llm_with_tools, chat_history)
        print(f"[AI]: {response.content}")


exports = {"main": main, "env": ["OPENAI_API_KEY"]}

if __name__ == "__main__":
    main()
