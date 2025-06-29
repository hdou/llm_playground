from retrieving import get_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | StrOutputParser()

retriever = get_retriever(n_results=3)

def generate_answer(question:str):
    docs = retriever.invoke(question)
    context = ""
    for doc in docs:
        context += doc.page_content
        for k, v in doc.metadata.items():
            context += f"{k}: {v}\n"
        context += "\n"
    answer = chain.invoke({"context": context, "question": question})
    return answer


if __name__ == "__main__":
    question = input("Enter a question (or 'q' to quit): ")
    while question != "q":
        answer = generate_answer(question)
        print(answer)
        question = input("Enter a question (or 'q' to quit): ")


