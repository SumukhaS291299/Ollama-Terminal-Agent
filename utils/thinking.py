from configparser import ConfigParser

import httpx
from langchain.messages import HumanMessage
from langchain_core.messages import ChatMessage
from langchain_ollama import ChatOllama

from .agentLogger import setup_logger
from .constants import API


class ThinkFist:
    def __init__(self, cfg: ConfigParser) -> None:
        self.logger = setup_logger("Thinking")
        self.conf = cfg
        self.parseConfig()
        assert self.healthCheck(), (
            f"Could not connect to ollama server:\t {self.Host}:{self.Port} or {self.ThinkModel} not found"
        )

    def parseConfig(self):
        self.ThinkModel = self.conf.get("Ollama", "ThinkingModel", fallback="granite4:3b")
        self.Scheme = self.conf.get("Ollama", "Scheme", fallback="http")
        self.Host = self.conf.get("Ollama", "Host", fallback="127.0.0.1")
        self.Port = int(self.conf.get("Ollama", "Port", fallback=11434))
        verifyStr = self.conf.get("Ollama", "Verify", fallback="no")
        self.Verify = False
        if verifyStr.lower() == "yes":
            self.Verify = True

    def healthCheck(self) -> bool:
        healthResp = httpx.get(
            url=f"{self.Scheme}://{self.Host}:{self.Port}{API.TAGS}", timeout=300
        )
        self.logger.info("Running health test...")
        self.logger.info(f"Running health:\t {healthResp.status_code}")
        healthDict = healthResp.json()
        for models in healthDict.get("models"):
            if self.ThinkModel == models.get("name"):
                return True
        return False

    def thinkContent(self):
        llm = ChatOllama(model=self.ThinkModel)
        # Will only work for granite
        # https://docs.langchain.com/oss/python/integrations/chat/ollama#reasoning-models-and-custom-message-roles
        # Will updat_set_config_context
        messages = [
            ChatMessage(role="control", content="thinking"),
            HumanMessage(""),
        ]

        response = llm.invoke(messages)
        print(response.content)
