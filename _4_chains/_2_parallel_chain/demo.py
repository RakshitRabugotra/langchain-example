import os

# LangChain chains/runnables
from langchain_core.runnables import RunnableParallel

# LangChain imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Hugging face model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login

# OpenAI model
from langchain_openai import ChatOpenAI

required_env = ["HUGGINGFACEHUB_ACCESS_TOKEN", "OPENAI_API_KEY", "OPENAI_MODEL"]

# Create a parser for this response
parser = StrOutputParser()

"""
What we want to do is:
1. We will prompt the hugging face model to create short notes on some text.
2. Simultaneously we will prompt the OpenAI model to create 5 short questions from the text
3. Lastly, we will ask the Huggingface model to merge both of the documents
"""

short_notes_prompt = PromptTemplate(
    template="Generate short and simple notes from the following text\n{text}",
    input_variables=["text"],
)

short_questions_prompt = PromptTemplate(
    template="Generate 5 short question answers from the following text\n{text}",
    input_variables=["text"],
)

# Merge request prompt
merge_request_prompt = PromptTemplate(
    template="Merge the provided notes and quiz into a single document\nnotes -> {notes}\nquestions -> {questions}",
    input_variables=["notes", "questions"],
)


def __create_openai_model():
    # Create the OpenAI model
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0.9)


def __create_huggingface_model():
    # Initialize the login,
    login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

    # Create a huggingface endpoint for the model
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        model_kwargs=dict(max_completion_tokens=200),
    )
    return ChatHuggingFace(llm=llm, temperature=0.9)


def __create_chain(huggingface_model, openai_model):
    # What we want to do is to create some parallel chains
    notes_chain = short_notes_prompt | huggingface_model | parser
    questions_chain = short_questions_prompt | openai_model | parser

    # Now, creating a parallel chain
    parallel_chain = RunnableParallel({
        "notes": notes_chain,
        "questions": questions_chain
    })

    # The final workflow would be to get the parallel chain outputs, and merge it
    merge_chain = merge_request_prompt | huggingface_model | parser
    return parallel_chain | merge_chain


def main():
    # Create the required models
    huggingface_model = __create_huggingface_model()
    openai_model = __create_openai_model()

    # Create our chain
    chain = __create_chain(huggingface_model, openai_model)

    # Invoke the chain with user input
    # user_input = input("-- your text --")
    user_input = \
    """
    In statistics, linear regression is a model that estimates the relationship between a scalar response (dependent variable) and one or more explanatory variables (regressor or independent variable). A model with exactly one explanatory variable is a simple linear regression; a model with two or more explanatory variables is a multiple linear regression.[1] This term is distinct from multivariate linear regression, which predicts multiple correlated dependent variables rather than a single dependent variable.[2]
    In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Most commonly, the conditional mean of the response given the values of the explanatory variables (or predictors) is assumed to be an affine function of those values; less commonly, the conditional median or some other quantile is used. Like all forms of regression analysis, linear regression focuses on the conditional probability distribution of the response given the values of the predictors, rather than on the joint probability distribution of all of these variables, which is the domain of multivariate analysis.
    """

    print("Generating response...")
    result = chain.invoke({
        "text": user_input
    })

    filename = "./out/output-parallel-chain.txt"
    print(f"The resultant document is stored in {filename}")
    with open(filename, mode='w+') as file:
        file.write(result)

    print("The resultant chain looks like: ")
    chain.get_graph().print_ascii()


# The files exports
exports = {"main": main, "env": required_env}

if __name__ == "__main__":
    main()
