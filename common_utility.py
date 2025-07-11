import os

# Saves the path if user says yes
def save_if_yes(content: str, save_directory: os.PathLike = "./out", ext: str = '.txt') -> None:

    # Create the file directory if it doesn't exist
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # Ask the user's choice
    choice = input("Do you want to save this to file? (Y/n): ")

    # Return only if negative
    if choice.lower() == 'n':
        return

    # Else ask the filename
    filename = os.path.join(save_directory, input("Enter the filename (without extension like .txt): ") + ext)
    with open(filename, mode='w+') as file:
        file.write(content)
    
    return print("Saved the file to " + filename)