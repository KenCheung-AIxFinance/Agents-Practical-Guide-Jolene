"""
https://python.langchain.com/docs/integrations/components/
https://medium.com/@mehulpratapsingh/langchain-agents-for-noobs-a-complete-practical-guide-e231b6c71a4a
"""

import getpass
import os
import dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

dotenv.load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter DeepSeek API Key: ")

from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.7)

# prompt = ChatPromptTemplate.from_template("你是一個計算機，負責計算數學問題，並返回準確的答案。指令：{query}")

@tool
def multiply(first_num: float, second_num: float) -> float:
    """Multiply to numbers together."""
    return first_num * second_num

@tool
def add(first_num: float, second_num: float) -> float:
    """add to numbers together."""
    return first_num + second_num


tools = [multiply, add]

llm_agent = create_react_agent(llm, tools)

input_message = {"role": "user", "content": "我有 7 排椅子，每排 8 张椅子，一共有多少张椅子 ?"}

response = llm_agent.invoke({"messages": [input_message]})

for message in response["messages"]:
    message.pretty_print()

