import os
from pydantic import BaseModel, Field
from typing import Optional

# LangChain imports
from langchain_openai import ChatOpenAI

required_env =["OPENAI_API_KEY", "OPENAI_MODEL"]

# defining a typed dict schema for the output
class Student(BaseModel):
    name: str = "Rakshit"
    age: Optional[int] = None
    cgpa: float = Field(gt=0, lt=10)


def main():
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0.9, max_completion_tokens=100)
    # Give the model a structured output response
    structured_model = model.with_structured_output(Student)

    print("Generating response...")
    # Now invoke the structured model, about a movie review
    response = structured_model.invoke(
        """The one we're talking about is a student named John Doe, who is 20 years old and has a CGPA of 8.5."""
    )

    if response and isinstance(response, Student):
      # If the response is a valid Student object, print its attributes
      print("Name:", response.name)
      print("Age:", response.age)
      print("CGPA:", response.cgpa)


exports = {
  'main': main,
  'env': required_env
}


if __name__ == "__main__":
    main()