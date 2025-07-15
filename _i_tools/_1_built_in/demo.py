from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool


def run_duckduckgo(search: str):
    return DuckDuckGoSearchRun().invoke(search)


def run_shelltool(command: str):
    return ShellTool().invoke(command)


def main():
    run_duckduckgo("What is the capital of France?")
    run_shelltool("ls -l")


exports = {
    'main': main,
    'env': None
}

if __name__ == "__main__":
    main() 