"""
Main entry point for the Dietary Consultant application.
"""
from langchain.agents.middleware.types import AgentState


from typing import Any


import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
import dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents.middleware import HumanInTheLoopMiddleware

# Local imports
from src.utils.workflow_utils import print_workflow_steps
from langchain.tools import tool


# Initialize environment and LLM
dotenv.load_dotenv()
print("Loading DeepSeek LLM...")
# print(f"Environment variables: {os.environ.get('DEEPSEEK_API_KEY')}")
llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.7)
# print(f"LLM initialized with API key: {llm.api_key}")


search_tool = DuckDuckGoSearchResults(output_format="list")

@tool()
def calculate_bmi(height_cm: float, weight_kg: float) -> str:
    """Calculate BMI based on height (cm) and weight (kg)."""
    try:
        height = float(height_cm) / 100
        weight = float(weight_kg)
        bmi = weight / (height ** 2)
        return f"BMI 為 {bmi:.2f}"
    except Exception as e:
        return f"錯誤：{str(e)}"

@tool
def get_runtime_datetime() -> dict:
    """
    Get current datetime from the runtime machine.
    """
    from datetime import datetime
    import socket
    import os

    now = datetime.now()

    return {
        "datetime": now.isoformat(),
        "hostname": socket.gethostname(),
        "runtime_id": os.getenv("RUNTIME_ID", "unknown")
    }


@tool()
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

# Initialize components
tools = [calculate_bmi, search_tool, get_runtime_datetime, send_email_tool]


hitl_middleware = HumanInTheLoopMiddleware[AgentState, None](
    interrupt_on={
        "send_email_tool": {
            "allowed_decisions": ["approve", "edit", "reject"],
        },
        "calculate_bmi":{
            "allowed_decisions": ["approve", "reject"]
        }
    }
)


def create_dietary_agent(
        llm,
        tools,
        checkpointer,
        debug: bool = False
):
    """Create a new agent with the given configuration.

    Args:
        llm: The language model to use
        tools: List of tools the agent can use
        checkpointer: Checkpoint saver for conversation history
        debug: Whether to enable debug mode

    Returns:
        The configured agent
    """
    agent_prompt = """你是一位親切的專業膳食營養顧問，專長於兒童與青少年營養。

你的任務：
1. 如果使用者尚未提供「年齡、性別、身高（cm）、體重（kg）」，請溫和地引導他們提供這些資訊。
   - 例如：「為了給您合適的建議，可以告訴我您的年齡、性別、身高和體重嗎？」
2. 一旦獲得足夠資訊，可主動計算 BMI（使用工具），並提供：
   - BMI 數值與兒童標準解讀（參考衛福部或 WHO 標準）
   - 每日建議熱量
   - 飲食與生活建議
3. 保持語氣友善、鼓勵，避免使用嚇人詞彙（如「肥胖」「過瘦」），改用「營養均衡」「健康成長」等。

請根據對話上下文決定是否需要工具協助。"""
    if (checkpointer == None):
        return create_agent(
        model=llm,
        tools=tools,
        system_prompt=agent_prompt,
        middleware=[hitl_middleware],
        debug=debug
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=agent_prompt,
        checkpointer=checkpointer,
        middleware=[hitl_middleware],
        debug=debug
    )

# agent = create_dietary_agent(llm=llm, tools=tools, checkpointer=InMemorySaver(), debug=True)

agent = create_dietary_agent(llm=llm, tools=tools, checkpointer=None, debug=True)
# config = get_agent_config()

if __name__ == '__main__':
    # ===== 優化後的互動主迴圈 =====
    # ===== 主互動迴圈（無預設問題）=====
    print("👋 歡迎使用兒童營養諮詢服務！")
    print("您可以輸入任何問題，例如：")
    print("  • 「我10歲，女生，身高138體重29」")
    print("  • 「怎麼判斷小孩體重是否正常？」")
    print("  • 「BMI 是什麼？」")
    print("輸入 'quit' 可隨時結束。\n")
    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in ["quit", "exit", "結束"]:
            print("👋 再見！")
            break

        # 傳送新訊息並取得完整執行軌跡
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            {"configurable": {"thread_id": 1}}
        )

        # 顯示完整 workflow 步驟（含推理、工具調用、結果）
        print_workflow_steps(response["messages"])
        pass

"""
1. 解釋代碼
2. 你寫代碼
3. 你畫圖去表達代碼的結構
4. 添加代碼：實際運行
"""

"""
TODO:
讓 AI Agent主動的查看並閱讀關於家人信息的文件
"""