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
    grade_decision: str


def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")


def get_json_llm():
    # Helper to get an LLM that is forced to output JSON if needed
    return ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")


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
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise."
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
        print("---GRADE: NO RELEVANT DOCUMENTS, QUEUE WEB SEARCH---")
        web_search = True

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
        "loop_count": loop_count,
    }


def transform_query(state: GraphState):
    """Transform the query to produce a better question."""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    loop_count = state.get("loop_count", 0)
    history = state.get("history", "") or ""

    system = (
        "You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. "
        "Look at the input and try to reason about the underlying semantic intent / meaning."
    )
    if history:
        system += "\nUse the conversation for context when needed.\nConversation so far:\n" + history

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print(f"Original: {question}\nRewritten: {better_question}")
    return {
        "documents": documents,
        "question": better_question,
        "loop_count": loop_count + 1,
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
    """Route question to web search or RAG."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    db = state.get("db")

    if not db:
        print("---NO KNOWLEDGE BASE, ROUTING TO WEB SEARCH---")
        return "web_search"

    llm = get_json_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at routing a user question to a vectorstore or web search. \n"
                   "The vectorstore contains documents the user uploaded and should be used for questions about their contents. \n"
                   "Only use web-search when the question is clearly about general knowledge, current events, or explicitly asks to search the web.\n"
                   "When in doubt, choose the vectorstore.\n"
                   "Return JSON with a single key 'datasource' and value 'web_search' or 'vectorstore'."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | JsonOutputParser()
    try:
        source = chain.invoke({"question": question})
        if source.get("datasource") == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
    except Exception:
        pass
    print("---ROUTE QUESTION TO RAG---")
    return "vectorstore"


def decide_to_generate(state: GraphState):
    """Determines whether to generate an answer, re-generate a question, or fall back to web search."""
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state.get("web_search", False)
    loop_count = state.get("loop_count", 0)

    if web_search:
        if loop_count >= 1:
            print("---NO RELEVANT DOCUMENTS, FALLING BACK TO WEB SEARCH---")
            return "web_search"
        print("---DECISION: ALL DOCUMENTS NOT RELEVANT, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_answer(state: GraphState):
    """Checks whether the generation is grounded in the documents and answers the question."""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    loop_count = state.get("loop_count", 0)

    if loop_count >= 2:
        print("---MAX LOOPS REACHED, ENDING---")
        return {"grade_decision": "useful", "loop_count": loop_count}

    llm = get_json_llm()
    # Hallucination Grader
    prompt_hallucination = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n"
                   "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. \n"
                   "Return JSON with a single key 'score' and value 'yes' or 'no'."),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    chain_hallucination = prompt_hallucination | llm | JsonOutputParser()

    # Answer Grader
    prompt_answer = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an answer addresses / resolves a question \n"
                   "Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question. \n"
                   "Return JSON with a single key 'score' and value 'yes' or 'no'."),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    chain_answer = prompt_answer | llm | JsonOutputParser()

    context = "\n\n".join([doc.page_content for doc in documents])

    try:
        score = chain_hallucination.invoke({"documents": context, "generation": generation})
        grade = score.get("score", "no")
        if grade.lower() == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            score_answer = chain_answer.invoke({"question": question, "generation": generation})
            grade_answer_val = score_answer.get("score", "no")
            if grade_answer_val.lower() == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return {"grade_decision": "useful", "loop_count": loop_count}
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return {"grade_decision": "not useful", "loop_count": loop_count}
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return {"grade_decision": "not supported", "loop_count": loop_count}
    except Exception:
        return {"grade_decision": "useful", "loop_count": loop_count}


def after_grade(state: GraphState):
    return state.get("grade_decision", "useful")


def create_agent_graph():
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("grade_answer", grade_answer)

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
            "transform_query": "transform_query",
            "generate": "generate",
            "web_search": "web_search",
        }
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "grade_answer")
    workflow.add_conditional_edges(
        "grade_answer",
        after_grade,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        }
    )

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
                yield ("token", chunk.content)
        elif mode == "updates":
            generate_state = data.get("generate")
            if generate_state and generate_state.get("documents"):
                final_docs = generate_state["documents"]

    yield ("done", final_docs)
