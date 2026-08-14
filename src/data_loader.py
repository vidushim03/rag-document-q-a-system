from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path):
    """
    Load a text-based PDF and return a list of LangChain Document objects.
    Falls back gracefully if a page has no text.
    """
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        # Filter out blank pages
        docs = [d for d in docs if d.page_content.strip()]

        print(f"Pages Loaded (with text): {len(docs)}")
        return docs

    except Exception as e:
        print(f"Error loading PDF '{path}': {e}")
        return []