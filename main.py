from module_import import get_import_map, Exports, option_name_formatter

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


def create_menu():
    """
    Create a menu to choose between the top-level directories
    """
    # Get all the options
    global_menu = get_import_map(debug=False)

    for module_name, module_tuple in global_menu.items():
        global_menu[module_name] = list(
            map(
                lambda entry: (
                    option_name_formatter(entry.name, module_name),
                    create_demo(entry.exports),
                ),
                module_tuple,
            )
        )
    
    # Now the global menu would like this:
    """
    {
        "_0_models": [
            ("Project", <function main at 0x000001FF625C8540>),
            ("Llms", <function main at 0x000001FF057CAFC0>),
            ("Chat Models", <function main at 0x000001FF08B7E660>),
            ("Embedding Models", <function main at 0x000001FF08B7EA20>),
        ],
        ...
        "_6_document_loaders": [("Text Loader", <function main at 0x000001FF0587E2A0>)],
    }
    """

    # Global menu would now be returned as a zip
    return list(global_menu.items())


def main():
    menu = create_menu()

    if menu:
        # Run the menu options
        option_title, menu_option = MenuLoader.run_options(
            menu, title="LangChain Examples", return_type="choice"
        )
        # Create a menu and launch it
        MenuLoader.run_options(
            menu_option, title=option_name_formatter(option_title, "")
        )


if __name__ == "__main__":
    main()
