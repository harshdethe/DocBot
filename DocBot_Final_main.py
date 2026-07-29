import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)
print(HF_TOKEN)
print(load_dotenv())

@st.cache_resource
def load_chain():
    # Loading LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )

    # custom prompt
    prompt = PromptTemplate(
        template="""
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.""",
        input_variables=["context", "question"],
    )

    # load dataset
    Db_Path = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    DB = FAISS.load_local(Db_Path, embedding_model, allow_dangerous_deserialization=True)

    # Retriever
    retriever = DB.as_retriever(search_kwargs={'k': 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")

st.title("💬 RAG Chatbot")
st.caption("Ask a question and get an answer grounded in your document context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input
user_question = st.chat_input("Ask your question here...")

if user_question:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # get answer from chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = load_chain()
                result = chain.invoke(user_question)
            except Exception as e:
                result = f"⚠️ Error: {e}"
            st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})