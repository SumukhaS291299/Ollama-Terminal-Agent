from configparser import ConfigParser

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from utils import setup_logger, config
from utils.messageBuilder import MessageBuilder
from utils.thinking import ThinkFirst


def main():
    setup_logger("Terminal Agent")

    conf: ConfigParser = config.Readconfig().read()
    thinking = ThinkFirst(conf)

    messages = MessageBuilder().build(SystemMessage(content="You should answer the question."),
                                      AIMessage(
                                          content="Sumukha S wrote terminal agent for ollama, he is a software engineer and developer"),
                                      HumanMessage(content="Who has coded terminal agent for ollama?"))

    response = thinking.think_content(messages=messages, think=True)
    print(response)


if __name__ == '__main__':
    main()
