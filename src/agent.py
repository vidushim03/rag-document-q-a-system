from typing import Any, Dict, List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.retriever import retrieve_docs


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    web_search: bool
    documents: List[Document]
    loop_count: int
    db: Any
    history: str


import os
import re


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text.strip()


def get_llm():
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(temperature=0, model_name=model_name)


def get_json_llm():
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(temperature=0, model_name=model_name)


def format_history(history):
    """Flatten a list of chat messages into a plain-text transcript."""
    if not history:
        return ""
    lines = []
    for message in history:
        role = message.get("role", "user")
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {message.get('content', '')}")
    return "\n".join(lines[-6:])


def retrieve(state: GraphState):
    """Retrieve documents from vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    loop_count = state.get("loop_count", 0)
    db = state.get("db")

    if not db:
        print("No knowledge base found. Falling back to web search.")
        return {"documents": [], "question": question, "web_search": True, "loop_count": loop_count}

    documents = retrieve_docs(db, question)
    return {"documents": documents, "question": question, "loop_count": loop_count}


def generate(state: GraphState):
    """Generate answer using RAG."""
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])
    loop_count = state.get("loop_count", 0)
    history = state.get("history", "") or ""

    system_parts = [
        "You are a helpful, friendly assistant. Use the following pieces of retrieved context to answer the user's question.\n"
        "- Answer based ONLY on the provided context when it contains relevant information.\n"
        "- Be conversational and approachable — avoid sounding robotic or stiff.\n"
        "- If the context doesn't cover the question, say so naturally and offer to help with something else.\n"
        "- You may use markdown formatting (bold, bullet points, code blocks) when it improves readability.\n"
        "- Do NOT start your response with <think> tags."
    ]
    if history:
        system_parts.append("Conversation so far:\n" + history)
    system_parts.append("Retrieved context:\n{context}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "\n\n".join(system_parts)),
        ("human", "{question}"),
    ])

    llm = get_llm()
    rag_chain = prompt | llm | StrOutputParser()

    context = "\n\n".join([doc.page_content for doc in documents])
    generation = rag_chain.invoke({"context": context, "question": question})
    generation = _strip_think(generation)
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "loop_count": loop_count + 1,
    }


def grade_documents(state: GraphState):
    """Determines whether the retrieved documents are relevant to the question (batch grading)."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    loop_count = state.get("loop_count", 0)

    llm = get_json_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing the relevance of retrieved documents to a user question. \n"
                   "A document is relevant if it contains keyword(s) or semantic meaning related to the question. \n"
                   "Provide the output as a JSON object with a single key 'scores'. "
                   "The value must be a dictionary mapping each document index (as a string) to either 'yes' or 'no'."),
        ("human", "User question: {question}\n\nRetrieved documents:\n{documents}"),
    ])
    chain = prompt | llm | JsonOutputParser()

    numbered = "\n\n".join(
        f"Document {i}:\n{d.page_content}" for i, d in enumerate(documents)
    )

    try:
        result = chain.invoke({"question": question, "documents": numbered})
        scores = result.get("scores", {}) or {}
    except Exception:
        print("---GRADE: BATCH GRADING FAILED, KEEPING ALL DOCUMENTS---")
        scores = {}

    filtered_docs = []
    web_search = False
    for i, d in enumerate(documents):
        grade = str(scores.get(str(i), "yes")).lower()
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    if not filtered_docs:
        print("---GRADE: NO DOC MARKED RELEVANT, KEEPING RETRIEVED DOCS---")
        filtered_docs = list(documents)
        web_search = False

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
        "loop_count": loop_count,
    }


def web_search(state: GraphState):
    """Web search based on the re-phrased question."""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    loop_count = state.get("loop_count", 0)

    try:
        tool = TavilySearchResults(max_results=3)
        docs = tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results_doc = Document(page_content=web_results, metadata={"source": "web_search"})
        documents.append(web_results_doc)
    except Exception as e:
        print(f"Web search failed (check TAVILY_API_KEY): {e}")

    return {"documents": documents, "question": question, "loop_count": loop_count}


# Routing logic
def route_question(state: GraphState):
    """Route question to web search or RAG.

    When the user has uploaded documents we always consult the vectorstore,
    so their files are never ignored.
    """
    print("---ROUTE QUESTION---")
    if not state.get("db"):
        print("---NO KNOWLEDGE BASE, ROUTING TO WEB SEARCH---")
        return "web_search"
    print("---ROUTE QUESTION TO RAG (documents uploaded)---")
    return "vectorstore"


def decide_to_generate(state: GraphState):
    """Determines whether to generate an answer from the documents or fall back to web search."""
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state.get("web_search", False)

    if web_search:
        print("---NO RELEVANT DOCUMENTS, FALLING BACK TO WEB SEARCH---")
        return "web_search"
    print("---DECISION: GENERATE---")
    return "generate"


def create_agent_graph():
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        }
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "web_search": "web_search",
        }
    )
    workflow.add_edge("generate", END)

    return workflow.compile()


def _build_inputs(question: str, db: Any, history) -> Dict[str, Any]:
    return {
        "question": question,
        "loop_count": 0,
        "db": db,
        "history": format_history(history),
    }


def run_agent(question: str, db: Any, history=None):
    app = create_agent_graph()
    inputs = _build_inputs(question, db, history)

    result = {}
    for output in app.stream(inputs):
        for key in output:
            print(f"Node '{key}':")
        if "generate" in output:
            result = output["generate"]
    return result


def stream_agent(question: str, db: Any, history=None):
    """Stream the final answer token-by-token.

    Yields ("token", text) tuples for the final generation and finally a
    ("done", documents) tuple carrying the documents the answer was based on.
    """
    app = create_agent_graph()
    inputs = _build_inputs(question, db, history)

    final_docs: List[Document] = []
    for mode, data in app.stream(inputs, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = data
            if meta.get("langgraph_node") == "generate" and chunk.content:
                yield ("token", _strip_think(chunk.content))
        elif mode == "updates":
            generate_state = data.get("generate")
            if generate_state and generate_state.get("documents"):
                final_docs = generate_state["documents"]

    yield ("done", final_docs)
