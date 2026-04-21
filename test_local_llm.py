from ml_engine.llm_agent import AutoMLChatAgent

agent = AutoMLChatAgent()
print(f"LLM Available: {agent.llm_available}")
res = agent.chat("What is your name?")
print(res)
