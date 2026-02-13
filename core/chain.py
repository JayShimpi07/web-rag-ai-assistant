from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def process_query(vectorstore, query):
    """
    Retrieves documents with scores and generates an answer.
    
    Args:
        vectorstore: The FAISS vectorstore.
        query: User question.
        
    Returns:
        dict: containing 'answer' and 'sources'.
    """
    # 1. Retrieve with scores
    # FAISS returns L2 distance by default (lower is better), 
    # but we can normalize or just display it. 
    # Usually strictly checking typical cosine similarity requires normalization_strategy in FAISS,
    # but here we just take the raw scores.
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    # 2. Format context
    context_text = "\n\n".join([doc.page_content for doc, _ in results])
    
    # 3. Setup Chain
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a strictly factual assistant. Use ONLY the provided context to answer the question.
    
    Rules:
    - If the answer cannot be verified from the provided context, you MUST say: "The answer cannot be verified from provided sources."
    - Do not fabricate information.
    - Keep the answer clear and concise.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    # 4. Generate Answer
    answer = chain.invoke({"context": context_text, "question": query})
    
    # 5. Format sources
    sources = []
    for doc, score in results:
        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)  # Convert numpy float to python float
        })
        
    return {
        "answer": answer,
        "sources": sources
    }
