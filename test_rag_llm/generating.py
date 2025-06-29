from retrieving import get_retriever, format_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# chain = prompt | llm | StrOutputParser()

retriever = get_retriever(n_results=3)

rag_chain = (
    {"context": retriever | format_context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def generate_answer(question:str):
    return rag_chain.invoke(question)

if __name__ == "__main__":
    question = input("Enter a question (or 'q' to quit): ")
    while question != "q":
        answer = generate_answer(question)
        print(answer)
        # print("-"*100)
        # prompt_hub_rag = hub.pull("rlm/rag-prompt")
        # print(prompt_hub_rag)
        question = input("Enter a question (or 'q' to quit): ")


