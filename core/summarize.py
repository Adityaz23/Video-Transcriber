import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a professional meeting analyst.

Summarize this transcript chunk.

Extract:
- Main discussion points
- Decisions made
- Action items
- Important deadlines

Use concise bullet points.
""",
            ),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunk_summarize = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summarize)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",  # FIX 1: was a bare string tuple ("system" role was missing), causing invalid prompt format
                """You are a senior business analyst.

Combine all partial summaries into a final report.

Structure:

# Overview

# Key Discussion Points

# Decisions Made

# Action Items

# Risks / Open Questions

Use professional bullet points.
Avoid repetition.
""",
            ),
            ("human", "{text}"),
        ]
    )
    combine_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )
    return combine_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()
    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "  # FIX 2: typo 'transccript'
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    return title_chain.invoke(transcript[:2000])
