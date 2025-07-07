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
ImportMap = dict[str, list[ModuleTuple]]


# Create a directory filter function
directory_filter = lambda files: list(
    filter(
        lambda file: os.path.isdir(file)
        and not file.startswith(".")
        and not file.startswith("__")
        and not file in [".git", "venv", ".venv", "res", "out"],
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
    # What we need to do is to import the `exports` from each module, and store it in the
    # dictionary
    for example in __import_map__:
        directories = __import_map__[example]
        # Now map the function to import specifically `exports`
        get_exports = lambda dir: (dir, importlib.import_module(dir).exports)
        # Update the import map to include 'demo.py' modules
        __import_map__[example] = list(
            map(get_exports, map(lambda _dir: _dir + ".demo", directories))
        )

    """
    Now the import map looks like this:
    {
        '_0_models': [
            ('_0_models.project.demo', {'main': <function main at 0x000001FEDFFDA200>, 'env': ['OPENAI_API_KEY', 'HUGGINGFACEHUB_ACCESS_TOKEN']}), 
            ...
            ('_0_models._2_embedding_models.demo', {'main': <function main at 0x000001FE8699A160>, 'env': ['OPENAI_API_KEY', 'HUGGINGFACEHUB_ACCESS_TOKEN']})
            ], 
        '_1_prompts': [
            ('_1_prompts._1_templates.demo', {'main': <function main at 0x000001FE86A11580>, 'env': None})
            ], 
        ...
        '_6_document_loaders': [
            ('_6_document_loaders._0_text_loader.demo', {'main': <function main at 0x000001FE86C263E0>, 'env': None})
            ]
        }
    """

    # now we safely return import_map
    import_map: ImportMap = {}
    for module_name, module_tuple in __import_map__.items():
        import_map[module_name] = list(
            map(lambda entry: ModuleTuple(entry[0], Exports(entry[1])), module_tuple)
        )

    if debug:
        print("[DEBUG]: Finished creating map for directories")

    return import_map
