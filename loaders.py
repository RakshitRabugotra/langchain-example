from dotenv import load_dotenv as load_dotenv
from typing import Literal as Literal, Callable, Union
import os


class EnvLoader:
    """
    Environment variable loader
    """

    # Load environment variables from .env file
    @staticmethod
    def load_env():
        return load_dotenv(dotenv_path="./.env")

    @staticmethod
    def verify_env(keys: list[str], check: Literal["all", "any"] = "any"):
        """
        :param keys: List of environment variable names to check
        :example: verify_env(['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'ANTHROPIC_API_KEY'])
        """
        failed = False
        reason = ""
        # Check if the environment variables are loaded
        for key in keys:
            if not os.getenv(key):
                reason = f"Environment variable {key} is not set."
                failed = True

            if failed and check == "any":
                failed = False
                # Warn the user if any variable is missing
                print("WARNING: " + reason + "\n")

            elif failed and check == "all":
                break

        if failed:
            raise EnvironmentError(reason)


# Menu driven option loader
class MenuLoader:
    """
    Menu driven option loader
    """

    @staticmethod
    def create_options_from_dict(
        _dict: dict,
        prefix: str,
        format_to_title: bool = True,
    ) -> list[tuple[str, Callable[[], None]]]:
        """
        Create options from `dict` items where the key starts with `prefix`
        :return: List of tuples containing option name and function to call
        """
        filter_func = lambda items: items[0].startswith(prefix)
        text_formatter = lambda option: (
            option[0].replace("_", " ").title() if format_to_title else option[0],
            option[1],
        )

        return list(
            map(
                text_formatter,  # Formats the function name to readable title
                filter(
                    filter_func, _dict.items()
                ),  # Filters functions that start with `prefix`
            )
        )

    @staticmethod
    def create_options_from_globals(
        globals_dict: dict,
        prefix: str,
        format_to_title: bool = True,
    ) -> list[tuple[str, Callable[[], None]]]:
        """
        Create options from globals() functions start with `prefix`
        :return: List of tuples containing option name and function to call
        """
        return MenuLoader.create_options_from_dict(
            globals_dict, prefix, format_to_title
        )

    @staticmethod
    def run_options(
        options: list[tuple[str, Callable[[], None]]],
        title: str = "Menu Loader",
        return_type: Literal["call", "chosen", "index"] = "call",
    ) -> Union[Callable[[], None], tuple[str, Callable[[], None]], int, None]:
        """
        :param options: List of tuples containing option name and function to call
        :example: load_options([("Option 1", func1), ("Option 2", func2)])
        """
        if not isinstance(options, list):
            # If the options is a map object, or filter object, convert it to list
            for obj_name in ["map", "filter"]:
                if obj_name in str(type(options)):
                    options = list(options)

        print(f"\n\nWelcome to the {title}!")
        print("Please select an option:")
        for i, (name, _) in enumerate(options, start=1):
            print(f"{i}. {name}")
        print("Or type '0' to quit.")

        while True:
            try:
                choice = int(input("\nEnter your choice: "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if choice == 0:
                exit(0)

            # Adjust choice to be zero-indexed
            choice -= 1

            if 0 > choice or choice > len(options):
                print("Invalid choice. Please try again.")
                continue

            # return according to the return type
            match return_type:
                case "call":
                    return options[choice][1]()
                case "chosen":
                    return options[choice]
                case "index":
                    return choice
            return None
