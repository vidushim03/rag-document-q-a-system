import os
import re
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


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text.strip()


class _ThinkingFilter:
    """Statefully strips <think>...</think> blocks from a token stream.

    The regex approach only works on a complete string, so per-token
    stripping lets the tags leak through. This filter buffers a few chars
    to detect markers that may span token boundaries.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"
    _OPEN_LEN = len(_OPEN)
    _CLOSE_LEN = len(_CLOSE)

    def __init__(self):
        self._pending = ""
        self._in_think = False

    def push(self, token: str):
        """Consume one token and yield any visible text it completes."""
        self._pending += token
        emitted = ""

        while True:
            if self._in_think:
                close = self._pending.find(self._CLOSE)
                if close == -1:
                    self._pending = self._pending[-self._CLOSE_LEN:]
                    return emitted
                self._pending = self._pending[close + self._CLOSE_LEN:]
                self._in_think = False
            
            open_idx = self._pending.find(self._OPEN)
            if open_idx == -1:
                keep = self._OPEN_LEN - 1
                if len(self._pending) > keep:
                    emitted += self._pending[:-keep]
                    self._pending = self._pending[-keep:]
                return emitted

            emitted += self._pending[:open_idx]
            self._pending = self._pending[open_idx + self._OPEN_LEN:]
            self._in_think = True

            close = self._pending.find(self._CLOSE)
            if close == -1:
                self._pending = self._pending[-self._CLOSE_LEN:]
                return emitted
            self._pending = self._pending[close + self._CLOSE_LEN:]
            self._in_think = False

    def flush(self) -> str:
        """Yield any remaining visible text at the end of the stream."""
        if self._in_think:
            self._pending = ""
        return self._pending


def get_llm():
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # Using 2048 to prevent hitting Groq's 8000 TPM limit 
    # (Requested tokens = Prompt tokens + max_tokens)
    return ChatGroq(temperature=0, model_name=model_name, streaming=True, max_tokens=2048)


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
    context = "\n\n".join([doc.page_content for doc in documents])

    if context:
        system_parts = [
            "You are a helpful assistant.",
            "Your primary source of truth is the Context below.",
            "Base your answer heavily on the provided context. If the context does not contain the complete answer, you may supplement it with your own knowledge.",
            "Never mention 'context', 'provided context', 'documents', or 'retrieved' in your answer.",
            "CRITICAL: Do NOT simply echo or repeat the user's question. Provide the requested summary or answer immediately.",
            "CRITICAL: If the user refers to an image, picture, or screenshot, DO NOT say you cannot view images. The images have already been processed by OCR and their text is included in the context. Answer based on that text.",
            "Be direct and concise. Use markdown when it improves readability.",
            "Context:\n{context}",
        ]
    else:
        system_parts = [
            "You are a helpful assistant. Answer directly from your own knowledge.",
            "CRITICAL: Do NOT simply echo or repeat the user's question. Provide the requested summary or answer immediately.",
            "Be concise. Use markdown when it improves readability.",
        ]
    if history:
        system_parts.append("Conversation so far:\n" + history)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "\n\n".join(system_parts)),
        ("human", "{question}"),
    ])

    llm = get_llm()
    rag_chain = prompt | llm | StrOutputParser()
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

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing the relevance of retrieved documents to a user question. \n"
                   "A document is relevant if it contains ANY keyword(s), entities, or semantic meaning related to the question. \n"
                   "When in doubt, grade it as 'yes'. \n"
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
        val = None
        if str(i) in scores:
            val = scores[str(i)]
        elif f"Document {i}" in scores:
            val = scores[f"Document {i}"]
        elif f"document {i}" in scores:
            val = scores[f"document {i}"]
            
        if val is None:
            grade = "yes"
        else:
            grade = str(val).lower()
            
        if "yes" in grade or "true" in grade:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    if not filtered_docs:
        print("---GRADE: NO DOC MARKED RELEVANT, GENERATING WITHOUT CONTEXT---")
        filtered_docs = []

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
    filter_ = _ThinkingFilter()
    for mode, data in app.stream(inputs, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = data
            if meta.get("langgraph_node") == "generate" and chunk.content:
                if chunk.__class__.__name__ == "AIMessageChunk" or getattr(chunk, "type", "") == "ai":
                    cleaned = filter_.push(chunk.content)
                    if cleaned:
                        yield ("token", cleaned)
        elif mode == "updates":
            generate_state = data.get("generate")
            if generate_state and generate_state.get("documents"):
                final_docs = generate_state["documents"]

    remaining = filter_.flush()
    if remaining:
        yield ("token", remaining)
    yield ("done", final_docs)
