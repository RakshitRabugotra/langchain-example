import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login

required_env = ["HUGGINGFACEHUB_ACCESS_TOKEN"]

"""
What we're trying to do:
1. We will prompt the LLM to make a report on some topic. Let's say about Black holes.
2. The LLM will generate a response, which is a string.
3. We want to again invoke the LLM with the generated response, and ask it to summarize the response in 5 lines.
4. The output will be a string, which is the summary of the response.
"""


def init_model():
    # Login using the env variables to huggingface
    login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

    # Create a huggingface endpoint for the model
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation"
    )
    return ChatHuggingFace(llm=llm, temperature=0.9, max_completion_tokens=100)


# Prompt Template to generate a detailed report
template1 = PromptTemplate(
    input_variables=["topic"],
    template="""
    Write a detailed report on the topic: {topic}.
    The report should be comprehensive and cover all aspects of the topic.
    The report should be in a formal tone and should be suitable for academic purposes.
    """,
)

# Prompt Template to summarize the report
template2 = PromptTemplate(
    input_variables=["report"],
    template="""
    Summarize the following report in 5 lines:
    {report}
    The summary should be concise and to the point.
    """,
)

# String output parser to parse the output of the model
parser = StrOutputParser()


def example_without_chains():
    print("Initializing the model...")
    model = init_model()
    print("Model initialized successfully.\n")

    # Draft the prompts
    prompt1 = template1.invoke({"topic": "Black holes"})
    # Invoke the model with the first prompt
    report = model.invoke(prompt1)
    print("Generated Report:\n", report.content)

    # Now we have the report, we can use it to generate a summary
    prompt2 = template2.invoke({"report": report.content})
    # Invoke the model with the second prompt
    summary = model.invoke(prompt2)
    print("\nSummary of the Report:\n", summary.content)


def example_with_chains():
    
    print("Initializing the model...")
    model = init_model()
    print("Model initialized successfully.\n")

    # Create a chain for the process
    chain = template1 | model | parser | template2 | model | parser

    # Invoke the chain with the topic
    topic = "Black holes"
    print("Generating report and summary for topic:", topic)
    result = chain.invoke({"topic": topic})

    print("\nSummary of the Report:\n", result)


def main():
    example_with_chains()

exports = {
    'main': main,
    'env': required_env,
}

if __name__ == "__main__":
    main()
