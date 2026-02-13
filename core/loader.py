import os
import tempfile
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document

def load_documents(source, source_type="url"):
    """
    Loads documents from various sources.
    
    Args:
        source: The input source. Can be a URL string, a file-like object (BytesIO), or a string of text.
        source_type: One of "url", "pdf", "txt", "csv", "text".
        
    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    documents = []

    try:
        if source_type == "url":
            # source is expected to be a string URL
            loader = WebBaseLoader(source)
            documents = loader.load()

        elif source_type == "text":
            # source is a raw text string
            documents = [Document(page_content=source, metadata={"source": "Direct Input"})]

        elif source_type in ["pdf", "txt", "csv"]:
            # source is expected to be a file-like object (BytesIO) from Streamlit
            # We need to write it to a temp file because many LangChain loaders require a file path
            
            suffix = f".{source_type}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(source.read())
                tmp_path = tmp_file.name

            try:
                if source_type == "pdf":
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                elif source_type == "txt":
                    loader = TextLoader(tmp_path, encoding="utf-8")
                    documents = loader.load()
                elif source_type == "csv":
                    loader = CSVLoader(tmp_path, encoding="utf-8")
                    documents = loader.load()
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    except Exception as e:
        print(f"Error loading {source_type}: {e}")
        # Optionally return an error document or raise
        pass

    return documents
