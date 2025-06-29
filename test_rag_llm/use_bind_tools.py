from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from retrieving import get_retriever, format_context

retriever = get_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def query_agenda_tool(query:str) -> str:
    """Query the agenda for the given query"""
    docs = retriever.invoke(query)
    return format_context(docs)

augmented_llm = llm.bind_tools([query_agenda_tool])

instruction = """
You are a helpful assistant that can answer questions about the agenda.
Use the query_agenda_tool to answer questions about agenda
"""

def generate_answer(question:str):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]
    answer = augmented_llm.invoke(messages)
    answer.pretty_print()
    return answer

if __name__ == "__main__":
    question = input("Enter a question (or 'q' to quit): ")
    while question != "q":
        answer = generate_answer(question)
        # print(answer)
        question = input("Enter a question (or 'q' to quit): ")
