import os
from configparser import ConfigParser

from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import AnyMessage

from utils import config, setup_logger
from utils.messageBuilder import MessageBuilder
from utils.thinking import ThinkFirst
from utils.tools import TerminalTools


def main():
    setup_logger("Terminal Agent")

    conf: ConfigParser = config.Readconfig().read()

    thinking = ThinkFirst(conf)
    messages: list[AnyMessage] = MessageBuilder().build(
        # The "Rulebook"
        SystemMessage(
            content=(
                "You are a terminal assistant. You must determine the OS and Shell "
                "from the conversation history and strictly follow those syntax rules."
                "You know runCommand tool which can run any command"
            )
        ),
        HumanMessage(content="I'm using a windows PC please give me commands for PWSH only."),
        # AIMessage(content="The user is on a Windows PC and I will use 'pwsh' for all commands."),
        HumanMessage(
            content="Find all python files recursively, sort them by file size descending, and show the top 5 files."
        ),
    )
    plan_text = thinking.think_content(messages=messages, temperature=0.6, think=False)
    print(plan_text)

    teminal_tool = TerminalTools(conf)

    # messages = MessageBuilder().build(
    #     SystemMessage(content="You are a helpful assistant, you are allowed to use any available tool"),
    #     HumanMessage(content="Welcome Sumukha S"),
    # )
    # teminal_tool.tool_content(messages=messages, temperature=0.1, think=False)

    # set_debug(True)

    messages_tool: list[AnyMessage] = MessageBuilder().build(
        *messages,  # keep original context
        AIMessage(content=f"Plan:\n{plan_text}"),  # assistant reasoning
        HumanMessage(content="Follow the Plan, use runCommand tool when required"),
    )

    os.environ["COMSPEC"] = '"C:\\Program Files\\PowerShell\\7\\pwsh.exe"'

    teminal_tool.tool_content(messages=messages_tool, temperature=0.1, think=False)


if __name__ == "__main__":
    main()
