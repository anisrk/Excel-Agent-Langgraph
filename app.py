import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
import os

# Load API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Load your Excel file
df = pd.read_excel("employee_data_100.xlsx")

# Create the agent
agent = create_pandas_dataframe_agent(llm, df, verbose=False)

# ---------------------------------------------------
# 🎨 UI Configuration (ADD THIS PART HERE)
# ---------------------------------------------------
st.set_page_config(page_title="LangGraph Excel Agent", page_icon="📈", layout="centered")

st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #F7FAFC;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #4A90E2;
        color: white;
        border-radius: 6px;
    }
    h1 {
        color: #2B6CB0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------

# Main app UI
st.title("📊 LangGraph Excel Agent")
st.write("Ask intelligent questions about your Excel data below 👇")

# User input
user_q = st.text_input("💬 Your question:", placeholder="e.g. Which department has the highest average salary?")

if st.button("Ask"):
    with st.spinner("🤖 Analyzing your data..."):
        result = agent.invoke(user_q)
        st.success("✅ Done!")
        st.subheader("🧠 Answer:")
        st.write(result["output"])
