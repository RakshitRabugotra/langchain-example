import os
# Document loaders
from langchain_community.document_loaders import WebBaseLoader

# An example URL
URL = "https://www.flipkart.com/apple-macbook-air-m2-8-gb-512-gb-ssd-mac-os-monterey-mlxx3hn-a/p/itmc2732c112aeb1?pid=COMGFB2GNWNN9DN8"

# Where the content will be saved
SAVE_PATH = "./out/web-txt"
if not os.path.isdir(SAVE_PATH):
  os.makedirs(SAVE_PATH)

def main():
  # Get the url from the user
  url = input("\nEnter a url (or press <ENTER/RETURN> for default url)\nurl: ") or URL

  # User may enter wrong information
  # Therefore a catch phrase
  try:
    loader = WebBaseLoader(url)

    documents = loader.load()
    print("Count of Docs fetched: ", len(documents))

    # Save the text content to a file
    filename = input("Enter a file name (without .txt) to save content: ")
    with open(os.path.join(SAVE_PATH, f"{filename}.txt"), mode='w+') as file:
      file.write("\n".join([doc.page_content for doc in documents]))
    
    print("Saved to: " + os.path.join(SAVE_PATH, filename).replace("\\", "/"))
  except Exception as e:
    print("[ERROR]: ", str(e))


exports = {
  'main': main,
  'env': None # Well, we do need `USER_AGENT` for bs4, but its optional
}