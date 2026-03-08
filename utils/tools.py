from configparser import ConfigParser

import httpx
from langchain.tools import tool
from langchain_core.messages import BaseMessage, ChatMessage
from langchain_core.messages.ai import AIMessage
from langchain_ollama import ChatOllama
from typing_extensions import List

from .agentLogger import setup_logger
from .constants import API


class TerminalTools:
    def __init__(self, cfg: ConfigParser) -> None:
        self.logger = setup_logger("Tools")
        self.conf = cfg
        self.parseConfig()
        assert self.healthCheck(), (
            f"Could not connect to ollama server:\t {self.Host}:{self.Port} or {self.ThinkModel} not found"
        )
        self.BaseTools = [TerminalTools.sayHello]
        self.toolMap = {"sayHello": TerminalTools.sayHello}

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
            if self.ToolModel == models.get("name"):
                return True
        return False

    def parseConfig(self):
        self.ToolModel = self.conf.get("Ollama", "ToolModel", fallback="granite4:3b")
        self.Scheme = self.conf.get("Ollama", "Scheme", fallback="http")
        self.Host = self.conf.get("Ollama", "Host", fallback="127.0.0.1")
        self.Port = int(self.conf.get("Ollama", "Port", fallback=11434))
        verifyStr = self.conf.get("Ollama", "Verify", fallback="no")
        self.Verify = False
        if verifyStr.lower() == "yes":
            self.Verify = True

    @tool
    @staticmethod
    def sayHello(user: str) -> str:
        """Greet user saying hello.

        Args:
            user: Name of the user
            type: str
        Returns:
            type: string
        """
        print("Called greeting tool...")
        return f"Hello {user}"

    def tool_content(
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
    ) -> None:
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

        model = model_name or self.ToolModel

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
        ).bind_tools(self.BaseTools)

        response = llm.invoke(messages)

        print(response)
        print(type(response))

        if not (response.tool_calls):
            self.logger.warning("No tool calls was found")
        if isinstance(response, AIMessage) and response.tool_calls:
            print(response.tool_calls)
            for tcs in response.tool_calls:
                if tcs.get("type") == "tool_call":
                    print(tcs.get(id))
                    print("Calling function: ", tcs.get("name"))
                    print("Calling function...")
                    try:
                        fn = tcs.get("name")
                        fn()
                    except Exception as e:
                        self.logger.error(e)
