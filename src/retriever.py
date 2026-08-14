def retrieve_docs(db, query):
    """
    Smart retriever that searches with multiple query variations
    to handle cases where the document uses different terminology.
    """

    # Generate related search terms for common academic queries
    query_lower = query.lower()

    extra_queries = []

    if any(word in query_lower for word in ["conclusion", "conclude", "concluded"]):
        extra_queries = ["conclusion", "summary", "discussion", "findings", "result", "outcome"]

    elif any(word in query_lower for word in ["introduction", "background"]):
        extra_queries = ["introduction", "background", "overview", "motivation"]

    elif any(word in query_lower for word in ["result", "finding", "outcome"]):
        extra_queries = ["results", "findings", "outcome", "experiment", "analysis"]

    elif any(word in query_lower for word in ["method", "approach", "methodology"]):
        extra_queries = ["methodology", "method", "approach", "technique", "algorithm"]

    elif any(word in query_lower for word in ["abstract", "summary", "summarize", "summarise"]):
        extra_queries = ["abstract", "summary", "overview", "introduction"]

    # Collect docs from main query + extra queries
    seen = set()
    all_docs = []

    # Main query — fetch more docs
    main_docs = db.similarity_search(query, k=5)
    for d in main_docs:
        key = d.page_content[:100]
        if key not in seen:
            seen.add(key)
            all_docs.append(d)

    # Extra semantic queries
    for q in extra_queries:
        try:
            extra_docs = db.similarity_search(q, k=3)
            for d in extra_docs:
                key = d.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(d)
        except:
            pass

    # Cap at 8 chunks to avoid token overflow
    all_docs = all_docs[:8]

    print(f"\n--- Retrieved {len(all_docs)} chunks ---")
    for i, d in enumerate(all_docs):
        print(f"\nChunk {i+1}:")
        print(d.page_content[:200])

    return all_docs