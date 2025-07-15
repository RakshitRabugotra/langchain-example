from langchain_openai import OpenAI

required_env = ["OPENAI_API_KEY"]

def main():
  llm = OpenAI(model='gpt-3.5-turbo')
  prompt = input("Prompt: ") # "What is the capital of France?"
  response = llm.invoke(prompt)
  print(response)


exports = {
  'main': main,
  'env': required_env
}


if __name__ == "__main__":
  main()