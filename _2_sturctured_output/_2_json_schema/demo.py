import os
import json

# LangChain imports
from langchain_openai import ChatOpenAI
# Custom imports
from loaders import EnvLoader

student_schema = None

with open("./res/schema.json", "r") as file:
    student_schema = json.load(file)

required_env = ["OPENAI_API_KEY", "OPENAI_MODEL"]

def main():
    # Print the schema for debugging
    print("Student Schema, that we're gonna follow:\n", student_schema, end='\n\n')

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0.9, max_completion_tokens=100)
    # Give the model a structured output response
    structured_model = model.with_structured_output(student_schema)

    print("Generating response...")
    # Now invoke the structured model, about a movie review
    response = structured_model.invoke(
        """The one we're talking about is a student named John Doe, who is 20 years old and has a CGPA of 8.5."""
    )

    if response and isinstance(response, dict):
      print("Name:", response["name"])
      print("Age:", response["age"])
      print("CGPA:", response["cgpa"])


exports = {
    'main': main,
    'env': required_env
}


if __name__ == "__main__":
    main()
