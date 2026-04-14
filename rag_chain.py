import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful academic tutor.

Rules:
1. Use ONLY the provided context from the uploaded study materials.
2. Do NOT use outside knowledge.
3. If the answer is not found in the context, respond with:
   "I dont have enough information in the uploaded materials."
4. When appropriate, explain the answer step by step.
"""

def build_rag_chain(vectorstore):

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
{SYSTEM_PROMPT}

Context:
{{context}}

Student Question:
{{question}}

Answer:
"""
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain


def ask(chain, question):

    result = chain.invoke({"query": question})

    answer = result.get("result", "")
    sources = result.get("source_documents", [])

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
    
    