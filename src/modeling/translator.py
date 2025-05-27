import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

try:
    # load environment variables from .env file (requires `python-dotenv`)
    

    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(prompt="Translator-Bot")
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

model = init_chat_model("llama3-8b-8192", model_provider="groq")

system_template = "Translate the following from {language1} into {language2}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
# prompt_template = ChatPromptTemplate.from_messages(
#     [SystemMessage(content = system_template), HumanMessage("{text}")]
# )
language1 = input("The text is in: ")
language2 = input("The text will be translated to: ")
text = input("Enter the text to translate: ")
prompt = prompt_template.invoke({"language1": language1, "language2": language2, "text": text})
response = model.invoke(prompt)
print(response.content)
# print(prompt)


