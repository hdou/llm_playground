from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import numpy as np

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def load_vectorstore(persist_path:Path):
    vectorstore = SKLearnVectorStore(
        persist_path=persist_path,
        embedding=embeddings,
        serializer="parquet",
    )
    return vectorstore

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

if __name__ == "__main__":
    persist_path = Path("agenda_rag_sklearn_vectorstore.parquet")
    vectorstore = load_vectorstore(persist_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Ask user for query - quit if user enters "q"
    while True:
        query = input("Enter a query (or 'q' to quit): ")
        if query == "q":
            break
        # Get relevant documents
        relevant_docs = retriever.invoke(query)
        print(f"Retrieved {len(relevant_docs)} relevant documents")

        for doc in relevant_docs:
            print(doc.page_content)
            print(doc.metadata)
            print("-"*100)

        print("="*100)
        print("Cosine similarity scores:")
        for doc in relevant_docs:
            # print the cosine similarity score
            query_embedding = embeddings.embed_query(query)
            answer_embedding = embeddings.embed_query(doc.page_content)
            similarity = cosine_similarity(query_embedding, answer_embedding)
            print(f"*** {similarity}")

