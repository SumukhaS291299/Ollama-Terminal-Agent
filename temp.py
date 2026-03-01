# import httpx

# r = httpx.get("https://www.httpbin.org/json", verify=False, timeout=300)
# print(r.status_code)
# print(r.json())

llm = ChatOllama(model=self.ThinkModel)
# Will only work for granite
# https://docs.langchain.com/oss/python/integrations/chat/ollama#reasoning-models-and-custom-message-roles
messages = [
    ChatMessage(role="control", content="thinking"),
    HumanMessage("What is 3^3?"),
]



response = llm.invoke(messages)
