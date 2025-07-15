from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Try to fetch the hub prompt
react_prompt = hub.pull("hwchase17/react")


def main():
    # Create a model
    llm = init_chat_model(model="gpt-4.1-mini", temperature=0.0)

    search_tool = DuckDuckGoSearchRun()

    # Create a react agent
    agent = create_react_agent(llm=llm, tools=[search_tool], prompt=react_prompt)

    # Create an agent executor
    agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

    # Run the agent
    query = input("[HUMAN]: ")

    response = agent_executor.invoke({"input": query})
    print(f"[AI]: {response['output']}")


exports = {"main": main, "env": ["OPENAI_API_KEY"]}
