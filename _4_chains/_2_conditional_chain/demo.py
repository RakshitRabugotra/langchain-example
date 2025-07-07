import os

# Type safety and structure
from typing import Optional, Literal
from pydantic import Field, BaseModel

# LangChain chains/runnables
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Output parsers
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

# LangChain imports
from langchain_core.prompts import PromptTemplate

# OpenAI model
from langchain_openai import ChatOpenAI

required_env = ["OPENAI_API_KEY", "OPENAI_MODEL"]


"""
What we want to do is:
1. We will prompt the hugging face model to analyze the sentiment of the customer feedback
2. Simultaneously we will prompt the OpenAI model to create 5 short questions from the text
3. Lastly, we will ask the Huggingface model to merge both of the documents
"""


"""Output Parsers"""
# Simple parser for the response
parser = StrOutputParser()

# We need a output parser for consistency
class FeedbackResponse(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the feedback classified to either of the possible enum values")
    sentiment_summary: Optional[str] = Field(description="Give the sentiment summary of the feedback in short sentence")
  
feedback_parser = PydanticOutputParser(pydantic_object=FeedbackResponse)

"""Prompt Templates"""

analyze_feedback_prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback text into 'positive' or 'negative'.\n{feedback}\n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={'format_instruction': feedback_parser.get_format_instructions()}
)

positive_reply_prompt = PromptTemplate(
    template="Write an appropriate response with joyful and appreciating tone to this positive feedback.\n{feedback}",
    input_variables=["feedback"],
)

negative_reply_prompt = PromptTemplate(
    template="Write an appropriate response with understanding and problem resolving tone to this negative feedback.\n{feedback}",
    input_variables=["feedback"],
)

# Merge request prompt
merge_request_prompt = PromptTemplate(
    template="Merge the provided notes and quiz into a single document\nnotes -> {notes}\nquestions -> {questions}",
    input_variables=["notes", "questions"],
)


def __create_openai_model():
    # Create the OpenAI model
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0.9)

def __create_chain(model):
    
    # We need a feedback analysis chain
    feedback_classifier_chain = analyze_feedback_prompt | model | feedback_parser

    # Now we will take a decision and branch
    branch_chain = RunnableBranch(
        # (condition1, chain1),
        # (condition2, chain2),
        # default chain

        # We will create lambda function which will return True, on certain conditions
        (lambda x: x.sentiment == 'positive', positive_reply_prompt | model | parser),
        (lambda x: x.sentiment == 'negative', negative_reply_prompt | model | parser),
        RunnableLambda(lambda x: "Could not find sentiment")
    )

    # Combine the final chain
    final_chain = feedback_classifier_chain | branch_chain
    return final_chain


def main():
    openai_model = __create_openai_model()

    # Create our chain
    chain = __create_chain(openai_model)

    # Invoke the chain with user input
    # user_input = input("-- your text --")
    user_input = \
    """
    The phone has decent hardware, but the software UI is a nightmare. 
    Everything feels clunky and unintuitive — even basic tasks take extra effort. 
    It's not worth the hassle. Wouldn't recommend unless you enjoy fighting with your phone.
    """

    print("Generating response...")
    result = chain.invoke({
        "feedback": user_input
    })

    filename = "./out/output-branch-chain.txt"
    print(f"The resultant document is stored in {filename}")
    with open(filename, mode='w+') as file:
        file.write(result)

    print("The resultant chain looks like: ")
    chain.get_graph().print_ascii()


# The files exports
exports = {"main": main, "env": required_env}

if __name__ == "__main__":
    main()
