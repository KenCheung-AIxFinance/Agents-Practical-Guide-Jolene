"""
https://python.langchain.com/docs/integrations/components/
https://medium.com/@mehulpratapsingh/langchain-agents-for-noobs-a-complete-practical-guide-e231b6c71a4a
"""

import getpass
import os
import dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter DeepSeek API Key: ")

from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.7)

prompt = ChatPromptTemplate.from_template("你是一個智能家居管家，負責接收用戶指令，並模擬返回結果。指令：{query}")

# StrOutputParser: 提取AIMessage 的 content
chain = prompt | llm | StrOutputParser()

response = chain.invoke({"query": "關閉空調"})
print(response)