import os

from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI

required_env =["OPENAI_API_KEY", "OPENAI_MODEL"]


# defining a typed dict schema for the output
class Review(TypedDict):
    """
    Simple TypedDict (but with Literals)
    """
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]

class AnnotatedReview(TypedDict):
    """
    Annotated TypedDict
    """
    summary: Annotated[str, "A structured output for movie reviews"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "Sentiment of the review"]

def main():
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0.9, max_completion_tokens=100)
    # Give the model a structured output response
    structured_model = model.with_structured_output(Review)

    print("Generating response...")
    # Now invoke the structured model, about a movie review
    response = structured_model.invoke(
        """
  Inception is a mind-bending thriller that keeps you on the edge of your seat
  with its intricate plot and stunning visuals. The performances are top-notch, especially Leonardo DiCaprio's portrayal of a troubled dream thief. The film's exploration of dreams within dreams is both fascinating and complex, making it a cinematic masterpiece that demands multiple viewings to fully appreciate its depth.
  """
    )

    print(response, type(response))
    if response:
      print("Summary:", response["summary"])
      print("Sentiment:", response["sentiment"])


exports = {
    'main': main,
    'env': required_env
}


if __name__ == "__main__":
    main()