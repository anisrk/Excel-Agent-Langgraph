
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
from colorama import Fore, Style, init
import pandas as pd
from IPython.display import Image, display
import os
import streamlit as st

init(autoreset=True)


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


llm = ChatOpenAI(model="gpt-4o", temperature=0)
print(Fore.GREEN + " OpenAI LLM initialized successfully!")

df = pd.read_excel("employee_data_100.xlsx")
print(Fore.GREEN + " Excel file loaded successfully!")
print(Fore.CYAN + " Preview of your dataset:")
print(df.head())

agent = create_pandas_dataframe_agent(llm, df, verbose=False)
print(Fore.GREEN + " Excel Agent created successfully!")


metadata = """
EmployeeID: Unique identifier for each employee.
Name: Full name of the employee.
Department: Department where the employee works.
Role: Job title or position.
Salary: Annual salary in USD.
JoiningDate: Date the employee joined the company.
City: City of the employee’s office location.
PerformanceRating: Rating from 1.0 (lowest) to 5.0 (highest).
"""
print(Fore.GREEN + " Custom metadata loaded successfully!")


class ExcelState(TypedDict):
    user_query: str
    metadata_context: str
    enriched_query: str
    result: str

def add_metadata(state: ExcelState):
    print(Fore.CYAN + "\n Step 1 → Attaching metadata...")
    state["metadata_context"] = metadata
    return state

def interpret_query(state: ExcelState):
    print(Fore.CYAN + " Step 2 → Interpreting user query with metadata...")
    q = state["user_query"]
    meta = state["metadata_context"]
    enriched = (
        f"You are analyzing employee data with the following metadata:\n{meta}\n\n"
        f"Answer the following question accurately:\n{q}"
    )
    state["enriched_query"] = enriched
    return state

def analyze_excel(state: ExcelState):
    print(Fore.CYAN + " Step 3 → Running Excel data analysis...")
    query = state["enriched_query"]
    try:
        result = agent.invoke(query)
        state["result"] = result
        print(Fore.GREEN + " Analysis completed successfully!")
    except Exception as e:
        state["result"] = f" Error: {e}"
        print(Fore.RED + f" Analysis failed: {e}")
    return state

graph = StateGraph(ExcelState)
graph.add_node("attach_metadata", add_metadata)
graph.add_node("interpret_query", interpret_query)
graph.add_node("analyze_excel", analyze_excel)

graph.set_entry_point("attach_metadata")
graph.add_edge("attach_metadata", "interpret_query")
graph.add_edge("interpret_query", "analyze_excel")
graph.add_edge("analyze_excel", END)

workflow = graph.compile()
print(Fore.GREEN + " LangGraph Excel Agent compiled successfully!")


try:
    png_data = workflow.get_graph(xray=True).draw_mermaid_png()
    with open("excel_agent_workflow.png", "wb") as f:
        f.write(png_data)
    display(Image(filename="excel_agent_workflow.png"))
    print(Fore.GREEN + " Workflow diagram saved as 'excel_agent_workflow.png'")
except Exception as e:
    print(Fore.YELLOW + f" Could not generate workflow diagram: {e}")


print(Fore.MAGENTA + "\n LangGraph Excel Agent is now LIVE!")
print(Fore.MAGENTA + "Type your questions below (or type 'exit' to quit):\n")

while True:
    user_q = input(Fore.YELLOW + " Your question: ")
    if user_q.lower().strip() == "exit":
        print(Fore.CYAN + "\n Exiting LangGraph Excel Agent. Goodbye!\n")
        break

print(Fore.BLUE + "\n Processing your question through LangGraph pipeline...\n")
initial_state = {"user_query": user_q}
result = workflow.invoke(initial_state)


answer_data = result.get("result", {})
if isinstance(answer_data, dict):
    final_answer = answer_data.get("output", "")
else:
    final_answer = str(answer_data)


final_answer = final_answer.replace("\\n", "\n").replace("\\t", "    ").strip()


###print(Fore.GREEN + "\n Final Answer:\n" + Style.RESET_ALL)
###print(Fore.WHITE + "---------------------------------------------")
###print(Fore.CYAN + final_answer)
###print(Fore.WHITE + "---------------------------------------------\n")
