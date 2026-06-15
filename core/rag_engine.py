import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from core.vector_store import build_vector_store, load_vector_store, retriever
from langchain_groq import ChatGroq


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = retriever(vector_store, k=4)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question based ONLY on the meeting transcript provided below.
                If the answer is not found in the context say:
                "I could not find this information in the meeting transcript."
                Always be concise and precise. If quoting someone, mention it clearly.
                Context from meeting transcript: {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )


# full LCEL rag chain =>
rag_chain = {"context": retriever | RunnableLambda(format_docs)}
