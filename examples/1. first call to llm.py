"""
https://python.langchain.com/docs/integrations/components/
https://medium.com/@mehulpratapsingh/langchain-agents-for-noobs-a-complete-practical-guide-e231b6c71a4a
"""

import getpass
import os
import dotenv

dotenv.load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter DeepSeek API Key: ")

from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.7)

response = llm.invoke("1+1等於多少")

print("response:", response.content)
