from typing import Type, Annotated, Any
import requests
import os

# Langchain imports
from langchain_core.tools import tool, BaseTool, InjectedToolArg
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage, BaseMessage


class NoConversionFoundException(Exception):
    """
    Raised when the conversion rate is not found
    """
    pass


def generate_endpoint(base_currency: str, target_currency: str) -> str:
    """
    Generates the endpoint for the currency conversion API
    Args:
        base_currency: The currency to convert from
        target_currency: The currency to convert to
    Returns:
        The endpoint
    """
    # Get the Api key
    API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
    
    if not API_KEY:
        raise EnvironmentError("EXCHANGE_RATE_API_KEY is not set")

    return f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{target_currency}"


# A tool to convert currency using API
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Gets the conversion factor for base currency to target currency
    Args:
        base_currency: The currency to convert from
        target_currency: The currency to convert to
    Returns:
        The conversion factor
    Raises:
        NoConversionFoundException if the conversion rate is not found
    NOTE: Only use this tool to convert currencies which are a means of monetary exchange. Not for any other purpose
    Example:
        get_conversion_factor("USD", "INR")
        # Returns: 82.5 (conversion factor)
    """
    # Create the endpoint
    url = generate_endpoint(base_currency, target_currency)
    # Make the request
    try:
        response = requests.get(url)
        # Throw an error if the request is not successful
        response.raise_for_status()
        # Get the data
        data = response.json()
        return data["conversion_rate"]
    except requests.exceptions.RequestException as e:
        raise NoConversionFoundException(f"Error fetching conversion rate: {e}")

    return amount


@tool
def convert(base_currency_value: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Converts the base currency value to the target currency value via conversion rate
    Args:
        base_currency_value: The currency to convert from
        conversion_rate: The conversion rate
    Returns:
        The converted amount
    Example:
        convert(100, 82.5)
        # Returns: 8250 (converted amount)
    """
    return base_currency_value * conversion_rate

# Our tool kit for Currency conversion
class CurrencyConversionToolkit:
    
    @staticmethod
    def get_tools() -> list[Type[BaseTool]]:
        return [get_conversion_factor, convert]


def repeat_calling_till_no_tool_calls(llm_with_tools: Any, chat_history: list[BaseMessage]) -> AIMessage:
    """
    Repeats calling the model till there are no tool calls
    """
    # Assume if don't know the conversion_rate
    conversion_rate = None
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
            # If we want to get the conversion factor
            if tool_call['name'] == "get_conversion_factor":
                print(f"[DEBUG] Getting conversion factor...")

                try:
                    conversion_factor_message = get_conversion_factor.invoke(tool_call)
                except NoConversionFoundException as e:
                    chat_history.append(ToolMessage(f"Error: No suitable conversion found for the requested units. Please check if it's a currency pair", tool_call_id=tool_call['id']))
                    return llm_with_tools.invoke(chat_history)

                conversion_rate = float(conversion_factor_message.content)
                # Append the conversion rate to the chat history
                chat_history.append(conversion_factor_message)

            # If we want to convert the amount
            if tool_call['name'] == "convert":
                print(f"[DEBUG] Converting amount...")
                # We have to modify the arguments of the tool call
                if not conversion_rate:
                    raise ValueError("Conversion rate is not known")
                tool_call['args']['conversion_rate'] = conversion_rate
                # Invoke the tool
                converted_amount_message = convert.invoke(tool_call)
                chat_history.append(converted_amount_message)


def main():

    # Create an llm
    llm = init_chat_model(model="openai:gpt-4.1-mini", temperature=0)
    # Create a chat model with tools
    # llm_with_tools = llm.bind_tools(CurrencyConversionToolkit.get_tools())
    llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

    # Create a chat history
    chat_history = [
        SystemMessage(
            """
            You are a helpful assistant that can convert currency. If you're asked to convert a currency to another, you will follow this plan of action:
            1. Get the conversion rate from the get_conversion_factor tool
            2. Use the conversion tool to convert the amount
            3. Return the converted amount.
            """
        )
    ]

    while True:
        query = input("[HUMAN]: ")

        if query.lower() == "exit":
            break

        # Append to chat history
        chat_history.append(HumanMessage(query))

        # Call the model, repeatedly till there are no tool calls
        response = repeat_calling_till_no_tool_calls(llm_with_tools, chat_history)
        print(f"[AI]: {response.content}")
        


exports = {"main": main, "env": ["OPENAI_API_KEY", "EXCHANGE_RATE_API_KEY"]}
