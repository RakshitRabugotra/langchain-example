from langchain.tools import StructuredTool
from pydantic import BaseModel, Field



def using_structured_tool():

    class MultiplyInput(BaseModel):
        a: float = Field(description="The first number to multiply")
        b: float = Field(description="The second number to multiply")

    def multiply(a: float, b: float) -> float:
        return a * b


    multiply_tool = StructuredTool.from_function(
        func=multiply,
        name="multiply",
        description="Multiply two numbers together",
        args_schema=MultiplyInput
    )

    return multiply_tool


def using_base_tool():
    
    class MultiplyTool(BaseTool):
        name: str = "multiply   "
        description: str = "Multiply two numbers together"
        args_schema: Type[BaseModel] = MultiplyInput

        def _run(self, a: float, b: float) -> float:
            return a * b
    
    return MultiplyTool()


def main():
    multiply_tool_1 = using_structured_tool()
    print(multiply_tool_1.invoke({"a": 2, "b": 3}))

    multiply_tool_2 = using_base_tool()
    print(multiply_tool_2.invoke({"a": 3, "b": 8}))


exports = {
    'main': main,
    'env': None
}

if __name__ == "__main__":
    main()