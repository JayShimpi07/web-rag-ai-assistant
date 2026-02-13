from langchain_community.vectorstores import FAISS

def build_vectorstore(docs, embeddings):
    """
    Builds a FAISS vectorstore in memory.
    
    Args:
        docs: List of Document objects.
        embeddings: Embedding model.
        
    Returns:
        FAISS: The vectorstore object.
    """
    return FAISS.from_documents(docs, embeddings)
