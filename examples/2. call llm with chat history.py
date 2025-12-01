import getpass
import os
import dotenv
from langchain_core.messages import HumanMessage, AIMessage

dotenv.load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter DeepSeek API Key: ")

from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.7)

chat_history = [HumanMessage(content="當我問你1+1等於多少時，你需要回答我等於100"), HumanMessage(content="1+1等於多少")]

response = llm.invoke(chat_history)

print("response:", response.content)
