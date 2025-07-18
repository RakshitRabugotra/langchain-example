import importlib
import os

# For typing
from typing import TypedDict, NamedTuple, Callable


# Create a typed dict object for exports
class Exports(TypedDict):
    main: Callable[[], None]
    env: list[str]


# The tuple you get as keys for the dict
class ModuleTuple(NamedTuple):
    name: str
    exports: Exports


# The type of the dict for import map
ImportMap = dict[str, list[str]]

# Some directories that are exempted from module_imports
exempted_directories = [".git", ".venv", "__pycache__", "tools", "out", "res"]

# Create a directory filter function
directory_filter = lambda files: list(
    filter(
        lambda file: os.path.isdir(file)
        and not file.startswith(".")
        and not file.startswith("__")
        and not file in exempted_directories,
        files,
    )
)

subdirectory_filter = lambda full_paths: list(
    filter(
        lambda full_path: os.path.isdir(full_path)
        and not os.path.split(full_path)[1].startswith(".")
        and not os.path.split(full_path)[1].startswith("__"),
        full_paths,
    )
)

# Function to format an option name from the directory structure
option_name_formatter: Callable[[str, str], str] = (
    lambda name, module_name: name.replace(module_name, "")
    .replace(".demo", "")
    .replace(".", "")
    .replace("_", " ")
    .title()
    # Ignoring the number before the option name
    [2:]
    .strip()
)

# Function to get the exports dictionary for a module
get_exports: Callable[[str], Exports] = lambda dir: Exports(
    # The module's export map which we store to Exports object
    importlib.import_module(dir + ".demo").exports
)


def get_import_map(debug=True):
    """
    Parses the modules and creates a `mapping` for each module to its subdirectories
    and `exports` to each subdirectory
    """
    if debug:
        print("[DEBUG]: Parsing the modules used in directories")

    """
    We want to search all the directories and make a mapping
    """
    __import_map__ = dict()

    # Get the top level directories
    top_level_directories = directory_filter(os.listdir())

    if debug:
        print("[DEBUG]: Top level directories: " + str(top_level_directories))

    # Iterate in each, and get the subdirectories containing the `demo.py` file
    for i, _dir in enumerate(top_level_directories, start=1):
        # Create full paths to test for directories
        full_paths = list(
            map(
                lambda inner_files: os.path.join(os.getcwd(), _dir, inner_files),
                os.listdir(_dir),
            )
        )
        # Update the import map
        subdirectories = subdirectory_filter(full_paths)

        # Now we need to omit the Current working directory in order to run modules
        # Also replace '\\' with '.' (windows) or '/' with '.' (not windows)
        is_windows = os.name == "nt"

        # Format the subdirectories for module friendly names
        subdirectories = list(
            map(
                lambda path: (
                    path.replace(os.getcwd(), "").replace("\\", ".")[1:]
                    if is_windows
                    else path.replace(os.getcwd(), "").replace("/", ".")
                ),
                subdirectories,
            )
        )
        # Update the import map
        __import_map__[_dir] = subdirectories

    # Now we will have structure like this
    """
    {
        "_0_models": [
            "_0_models.project",
            "_0_models._0_llms",
            "_0_models._1_chat_models",
            "_0_models._2_embedding_models",
        ],
        "_1_prompts": ["_1_prompts._1_templates"],
        "_2_sturctured_output": [
            "_2_sturctured_output._0_typed_dict",
            "_2_sturctured_output._1_pydantic",
            "_2_sturctured_output._2_json_schema",
        ],
        "_3_output_parser": [
            "_3_output_parser._0_stroutputparser",
            "_3_output_parser._1_jsonoutputparser",
            "_3_output_parser._2_structuredoutputparser",
            "_3_output_parser._3_pydanticoutputparser",
        ],
        "_4_chains": [
            "_4_chains._0_sequential_chain",
            "_4_chains._1_parallel_chain",
            "_4_chains._2_conditional_chain",
        ],
        "_5_runnables": ["_5_runnables._0_pdf_reader", "_5_runnables._1_retrieval_chain"],
        "_6_document_loaders": ["_6_document_loaders._0_text_loader"],
    }
    """

    if debug:
        print("[DEBUG]: Finished creating map for directories")

    return __import_map__
