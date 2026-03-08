from configparser import ConfigParser

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from utils import config, setup_logger
from utils.messageBuilder import MessageBuilder
from utils.thinking import ThinkFirst
from utils.tools import TerminalTools


def main():
    setup_logger("Terminal Agent")

    conf: ConfigParser = config.Readconfig().read()

    # thinking = ThinkFirst(conf)
    # messages = MessageBuilder().build(
    #     SystemMessage(content="You are a helpful assistant."),
    #     HumanMessage(content="Who wrote terminal agent for ollama?"),
    #     AIMessage(content="Sumukha S wrote terminal agent for ollama."),
    #     HumanMessage(content="Who coded it?"),
    # )
    # response = thinking.think_content(messages=messages, temperature=0.6, think=False)
    # print(response)

    teminal_tool = TerminalTools(conf)

    messages = MessageBuilder().build(
        SystemMessage(
            content="You are a helpful assistant, you are allowed to use any available tool"
        ),
        HumanMessage(content="Greet Sumukha S"),
    )
    teminal_tool.tool_content(messages=messages, temperature=0.1, think=False)


if __name__ == "__main__":
    main()
