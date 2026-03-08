# import httpx
import logging

from langchain_core.messages import ChatMessage, HumanMessage
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.DEBUG)

# r = httpx.get("https://www.httpbin.org/json", verify=False, timeout=300)
# print(r.status_code)
# print(r.json())

llm = ChatOllama(model="llama3.2:latest", base_url="http://192.168.29.11:11434")
# Will only work for granite
# https://docs.langchain.com/oss/python/integrations/chat/ollama#reasoning-models-and-custom-message-roles
messages = [
    ChatMessage(role="assistant", content="Hi, I'm Sumukha S and my age is 66 years"),
    HumanMessage("What is Sumukha's age ?"),
]
response = llm.invoke(messages)

print(response.content)
