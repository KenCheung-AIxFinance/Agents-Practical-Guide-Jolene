def print_workflow_steps(messages):
    """可視化 LangGraph ReAct agent 的推理與工具調用流程"""
    print("\n" + "=" * 60)
    print("🔍 Workflow 執行流程（節點步驟）")
    print("=" * 60)

    for i, msg in enumerate(messages):
        if msg.type == "human":
            print(f"[Step {i + 1}] 🧑 使用者輸入:")
            print(f"    {msg.content}\n")

        elif msg.type == "ai":
            # 檢查是否有 tool_calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"[Step {i + 1}] 🤖 AI 推理節點（決定調用工具）:")
                print(f"    思考: {msg.content or '(無額外說明)'}")
                for tc in msg.tool_calls:
                    print(f"    🔧 準備調用工具: {tc['name']}({tc['args']})")
                print()
            else:
                print(f"[Step {i + 1}] 🤖 AI 最終回應:")
                print(f"    {msg.content}\n")

        elif msg.type == "tool":
            print(f"[Step {i + 1}] 🛠️ 工具執行節點:")
            print(f"    工具: {msg.name}")
            print(f"    輸入: {msg.tool_call_id} | 參數已傳遞")
            print(f"    輸出: {msg.content}\n")

        else:
            print(f"[Step {i + 1}] ❓ 未知訊息類型 ({msg.type}): {msg}")