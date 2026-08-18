from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
import csv

def load_pdf(path):
    """
    Load a text-based PDF and return a list of LangChain Document objects.
    Falls back gracefully if a page has no text.
    """
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        # Filter out blank pages
        valid_docs = []
        for d in docs:
            if d.page_content.strip():
                if "page" in d.metadata:
                    d.metadata["page"] += 1
                valid_docs.append(d)
        docs = valid_docs

        print(f"Pages Loaded (with text): {len(docs)}")
        return docs

    except Exception as e:
        print(f"Error loading PDF '{path}': {e}")
        return []

def load_docx(path):
    try:
        loader = Docx2txtLoader(path)
        docs = loader.load()
        return [d for d in docs if d.page_content.strip()]
    except Exception as e:
        print(f"Error loading DOCX '{path}': {e}")
        return []

def load_txt(path):
    try:
        loader = TextLoader(path, encoding='utf-8')
        docs = loader.load()
        return [d for d in docs if d.page_content.strip()]
    except Exception as e:
        print(f"Error loading TXT '{path}': {e}")
        return []

def load_csv(path):
    try:
        loader = CSVLoader(path, encoding='utf-8')
        docs = loader.load()
        return [d for d in docs if d.page_content.strip()]
    except Exception as e:
        print(f"Error loading CSV '{path}': {e}")
        return []