from configparser import ConfigParser
from typing import List, Any

import httpx
import langchain_core.messages
from langchain.messages import HumanMessage
from langchain_core.messages import BaseMessage, ChatMessage

from langchain_ollama import ChatOllama

from .agentLogger import setup_logger
from .constants import API


class ThinkFirst:
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
            url=f"{self.Scheme}://{self.Host}:{self.Port}{API.TAGS.value}", timeout=300
        )
        self.logger.info("Running health test...")
        self.logger.info(f"Status code:\t {healthResp.status_code}")
        self.logger.debug(f"Health:\t {healthResp.status_code}")
        healthDict = healthResp.json()
        self.logger.debug(f"Health:\t {healthDict}")
        for models in healthDict.get("models"):
            if self.ThinkModel == models.get("name"):
                return True
        return False

    def think_content(
            self,
            messages: List[BaseMessage],
            model_name: str | None = None,
            temperature: float = 0.6,
            think: bool = False,
            **kwargs
    ) -> str | list[Any]:
        """
        Generic LLM invocation wrapper for Ollama models.
        Supports:
        - prompt
        - messages
        - conversation history
        """

        model = model_name or self.ThinkModel

        if think:
            messages.append(ChatMessage(role="control", content="thinking"))

        base_url = f"{self.Scheme}://{self.Host}:{self.Port}"

        llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            # format="json",
            reasoning=think,
            **kwargs
        )

        llm.format = "json"

        response = llm.invoke(messages)

        # print("Resp:", response) TODO Remove DEBUG print
        thinkContent = response.additional_kwargs.get("reasoning_content")
        return response.content
