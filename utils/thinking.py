from configparser import ConfigParser
from typing import Any, List

import httpx
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
        format: str | dict | None = None,
        num_predict=256,
        num_gpu: int | None = None,
        repeat_penalty: float | None = None,
        top_k=40,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str | list[Any]:
        """
        Generate a response from the language model based on the provided messages.

            This method sends a conversation history to the model and returns the
            generated output. Various generation parameters can be adjusted to control
            creativity, response length, and sampling behavior.

            Args:
                messages (List[BaseMessage]):
                    Conversation messages to send to the model.

                model_name (str | None, optional):
                    Name of the model to use. If None, the default configured model
                    will be used.

                temperature (float, optional):
                    Controls randomness of generation. Lower values produce more
                    deterministic responses. Default is 0.6.

                think (bool, optional):
                    Enables reasoning or "thinking" mode if supported by the model.

                format (str | None, optional):
                    Optional response format (for example: "json").

                num_predict (int, optional):
                    Maximum number of tokens to generate in the response.

                num_gpu (int | None, optional):
                    The number of GPUs to use.

                    It defaults to `1` to enable metal support, `0` to disable.

                repeat_penalty (float | None, optional):
                    Penalizes repeated tokens to reduce repetitive outputs.

                top_k (int | None, optional):
                    Reduces the probability of generating nonsense.

                    A higher value (e.g. `100`) will give more diverse answers, while a lower value
                    (e.g. `10`) will be more conservative.

                    (Default: `40`)

                 stop: list[str] | None = None
                    Sets the stop tokens to use.

                **kwargs:
                    Additional model-specific parameters passed through to the backend.

            Returns:
                str | list[Any]:
                    The generated response. May return a string or a structured
                    response depending on the selected format.
        """

        model = model_name or self.ThinkModel

        if think:
            messages.append(ChatMessage(role="control", content="thinking"))

        base_url = f"{self.Scheme}://{self.Host}:{self.Port}"

        llm = ChatOllama(
            validate_model_on_init=True,
            model=model,
            temperature=temperature,
            base_url=base_url,
            format=format,
            reasoning=think,
            num_predict=num_predict,
            num_gpu=num_gpu,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stop=stop,
            **kwargs,
        )

        response = llm.invoke(messages)

        # print("Resp:", response) TODO Remove DEBUG print
        # thinkContent = response.additional_kwargs.get("reasoning_content")
        return response.content
