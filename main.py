from module_import import get_import_map, get_exports, option_name_formatter, Exports

# Custom modules for environment loading
# and to create menu
from loaders import MenuLoader, EnvLoader


def create_demo(exports: Exports):
    # Load the environment variables
    EnvLoader.load_env()

    # Verify that all required env variables are loaded
    if exports.get("env"):
        EnvLoader.verify_env(exports["env"], check="all")

    # Call the main function
    return exports["main"]


def format_import_map():
    """
    Create a menu to choose between the top-level directories
    """
    # Get all the options
    global_menu = get_import_map(debug=False)
    # Global menu would now be returned as a zip
    return list(global_menu.items())


def main():
    menu = format_import_map()

    if not menu:
        exit(-1)

    # Run the menu options
    choice_index = MenuLoader.run_options(
        map(
            lambda x: (option_name_formatter(x[0], ""), x[1]), menu
        ),  # Change the format of displayed menu
        title="LangChain Examples",
        return_type="index",
    )

    # Extract the option title
    option_title, runnable_module_paths = menu[choice_index]

    # Create a menu and launch it
    _, demo_module_path = MenuLoader.run_options(
        [
            (option_name_formatter(module, option_title), module)
            for module in runnable_module_paths
        ],
        title=option_name_formatter(option_title, ""),
        return_type="chosen",
    )
    # we will import the runnable module
    exports = get_exports(demo_module_path)
    # Create the demo and launch it
    create_demo(exports)()


if __name__ == "__main__":
    main()
