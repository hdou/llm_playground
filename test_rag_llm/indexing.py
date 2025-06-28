import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from pathlib import Path

from agenda_parser import AgendaParser

def count_tokens(text, model="cl100k_base"):
    """
    Count the number of tokens in the text using tiktoken.

    Args:
        text (str): The text to count tokens for
        model (str): The tokenizer model to use (default: cl100k_base for GPT-4)

    Returns:
        int: Number of tokens in the text
    """
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

class AgendaDocumentLoader:
    def load(self):
        with open("agenda-pdf.txt", "r", encoding="utf-8") as f:
            text = f.read()
        parser = AgendaParser(text)
        fragments = parser.parse()
        documents = [self.fragment_to_document(fragment) for fragment in fragments]
        return documents

    def fragment_to_document(self, fragment):
        return Document(
            page_content=fragment.content,
            metadata={
                "time": fragment.time,
                "location": fragment.location
            }
        )

# def load_agenda_fragments():
#     with open("agenda-pdf.txt", "r", encoding="utf-8") as f:
#         text = f.read()
#     parser = AgendaParser(text)
#     fragments = parser.parse()
#     return fragments

# def merge_fragment(frag):
#     """
#     Combine all keys and contents of the fragment into a single string,
#     as time:<time>\nlocation:<location>\ncontent:<content>
#     """
#     return f"time:{frag.time}\nlocation:{frag.location}\ncontent:{frag.content}"

# def load_agenda_docs():
#     fragments = load_agenda_fragments()
#     docs = []
#     for frag in fragments:
#         docs.append(merge_fragment(frag))
#     return docs

def split_documents(documents):
    """
    Split documents into smaller chunks for improved retrieval.

    This function:
    1. Uses RecursiveCharacterTextSplitter with tiktoken to create semantically meaningful chunks
    2. Ensures chunks are appropriately sized for embedding and retrieval
    3. Counts the resulting chunks and their total tokens

    Args:
        documents (list): List of Document objects to split

    Returns:
        list: A list of split Document objects
    """
    print("Splitting documents...")

    # Initialize text splitter using tiktoken for accurate token counting
    # chunk_size=8,000 creates relatively large chunks for comprehensive context
    # chunk_overlap=500 ensures continuity between chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000,
        chunk_overlap=500
    )

    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents)

    print(f"Created {len(split_docs)} chunks from documents.")

    # Count total tokens in split documents
    total_tokens = 0
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content)

    print(f"Total tokens in split documents: {total_tokens}")

    return split_docs

def create_vectorstore(splits, persist_path:Path):
    """
    Create a vector store from document chunks using SKLearnVectorStore.

    This function:
    1. Initializes an embedding model to convert text into vector representations
    2. Creates a vector store from the document chunks

    Args:
        splits (list): List of split Document objects to embed
        persist_path (Path): Path to save the vector store
    Returns:
        SKLearnVectorStore: A vector store containing the embedded documents
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if persist_path.exists():
        vectorstore = SKLearnVectorStore(
            persist_path=persist_path,
            embedding=embeddings,
            serializer="parquet",
        )
        print(f"Loading vector store from {persist_path}")
        return vectorstore

    print("Creating SKLearnVectorStore...")

    # Create vector store from documents using SKLearn
    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=persist_path   ,
        serializer="parquet",
    )
    print("SKLearnVectorStore created successfully.")

    vectorstore.persist()
    print("SKLearnVectorStore was persisted to", persist_path)

    return vectorstore

if __name__ == "__main__":
    loader = AgendaDocumentLoader()
    documents = loader.load()
    split_docs = split_documents(documents)

    persist_path = Path("agenda_rag_sklearn_vectorstore.parquet")
    vectorstore = create_vectorstore(split_docs, persist_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    query = "Find the agenda items about machine vision"
    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")

    for doc in relevant_docs:
        print(doc.page_content)
        print(doc.metadata)
        print("-"*100)