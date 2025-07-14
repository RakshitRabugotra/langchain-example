from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers together.
    """
    return a * b


def main():
    
    # The tools definition contains
    print(multiply.name)
    print(multiply.description
    )
    print(multiply.args)

    # The tool can be used like any other tool
    print(multiply.invoke(
        {"a": 2, "b": 3}
    ))

    # The JSON schema of the tools
    print(multiply.args_schema.model_json_schema())


exports = {
    'main': main,
    'env': None
}

if __name__ == "__main__":
    main()