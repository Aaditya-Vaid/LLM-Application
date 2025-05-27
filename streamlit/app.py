import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# Load .env variables
load_dotenv()

# Set environment variables via UI if not set
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.sidebar.text_input(
        "Enter Groq API Key", type="password"
    )

if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = st.sidebar.text_input(
        "LangSmith API Key (optional)", type="password"
    )

if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = st.sidebar.text_input(
        "LangSmith Project", value="Translator-Bot"
    )

os.environ["LANGSMITH_TRACING"] = "true"


# Load the model once
@st.cache_resource
def load_model():
    return init_chat_model("llama3-8b-8192", model_provider="groq")


model = load_model()

# Prompt template
system_template = "Translate the following from {language1} into {language2}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Chatbot UI
st.title("Translator Chatbot")
st.caption("Powered by Groq + LLaMA3-8B")

# Language selector
language1 = st.sidebar.selectbox(
    "Select text language",
    ["Spanish", "French", "German", "Hindi", "Japanese", "Chinese", "Arabic", "Korean"],
)
language2 = st.sidebar.selectbox(
    "Select target language",
    ["Spanish", "French", "German", "Hindi", "Japanese", "Chinese", "Arabic", "Korean"],
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
if user_input := st.chat_input("Type text to translate..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build and invoke model prompt
    prompt = prompt_template.invoke({"language1": language1, "language2": language2, "text": user_input})
    response = model.invoke(prompt)

    # Append response to chat history
    bot_reply = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
