import json
import os
from configparser import ConfigParser
from re import S
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import AnyMessage
from rich.ansi import AnsiDecoder
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static, TextArea

from ota_utils import config, setup_logger
from ota_utils.messageBuilder import MessageBuilder
from ota_utils.thinking import ThinkFirst
from ota_utils.tools import TerminalTools

logger = setup_logger("Terminal Agent")


# ------------------------------
# Core agent logic
# ------------------------------
def run_agent(task: str):

    conf: ConfigParser = config.Readconfig().read()

    os.environ["SHELL_TYPE"] = conf.get("Shell", "type", fallback="PWSH").upper()
    SHELL = os.environ["SHELL_TYPE"]
    logger.info(f"Shell used : {SHELL}")

    thinking = ThinkFirst(conf)

    # Use default message block as a fallback
    messages: list[AnyMessage] = MessageBuilder().build(
        SystemMessage(
            content=(
                "You are a terminal assistant. Determine the OS and Shell "
                "from the conversation history and follow those syntax rules. "
                "Use the runCommand tool whenever execution is required."
            )
        ),
        HumanMessage(content=f"I'm {SHELL} shell type and you have to give me commands that will work with it."),
        HumanMessage(content=task),
    )

    try:
        if conf.has_option("Prompt", "BuilderFile"):
            builder_file = conf.get("Prompt", "BuilderFile")
            with open(builder_file, "r") as builderFile:
                builderMessageDict = json.load(builderFile)
            systemMessages: list[str] = builderMessageDict["system"]
            humanMessages: list[str] = builderMessageDict["human"]
            if len(systemMessages) == 0 and len(humanMessages) == 0:
                raise Exception("Nothing in file")
            if len(systemMessages) == 0 and len(humanMessages) > 0:
                systemMessages.append("""You are a terminal assistant.
                    Determine the OS and Shell from the conversation history and follow those syntax rules.
                    Use the runCommand tool whenever execution is required.""")
                logger.info("Using prompts ")
                logger.info(f"[System]: {systemMessages}")
                logger.info(f"[Human]: {humanMessages}")
            messages: list[AnyMessage] = MessageBuilder().build(
                SystemMessage(content=(" ".join(systemMessages))),
                HumanMessage(content=" ".join(humanMessages)),
                HumanMessage(content=task),
            )
    except Exception as e:
        logger.error("Failed to load file ", e)

    plan_text = thinking.think_content(messages=messages)

    terminal_tool = TerminalTools(conf)

    messages_tool: list[AnyMessage] = MessageBuilder().build(
        *messages,
        AIMessage(content=f"Plan:\n{plan_text}"),
        HumanMessage(content="Execute the plan using runCommand."),
    )

    result = terminal_tool.tool_content(messages=messages_tool)

    return plan_text, result


# ------------------------------
# CLI mode
# ------------------------------
def main():
    task = input("Enter task: ")
    plan, result = run_agent(task)
    print("\nPLAN:\n", plan)
    print("\nRESULT:\n", result)
    print("\nRESULT:\n", result.get("stdout"))
    if result.get("stderr"):
        print("This error is mostly coming from a wrong command from your LLM")
        print("\nERROR:\n", result.get("stderr"))


# ------------------------------
# TUI Application
# ------------------------------
class TerminalAgentApp(App):
    BINDINGS = [Binding("ctrl+c", "copy_output", "Copy Output")]
    CSS = """
    Screen {
        layout: vertical;
    }

    #status_bar {
        height:3;
        border: round green;
    }

    #plan_panel {
        height:10;
        border: round yellow;
    }

    #metrics_panel {
        height:5;
        border: round cyan;
    }

    #output_panel {
        height:1fr;
        border: round magenta;
    }

    Input {
        dock: bottom;
    }
    RichLog {
        overflow-y: auto;
    }
    """
    # ------------------------------

    def compose(self) -> ComposeResult:

        yield Header(show_clock=True)

        with Vertical(id="main_area"):
            # STATUS BAR
            with Horizontal(id="status_bar"):
                yield Static(" PLAN: idle ", id="status_plan")
                yield Static(" EXEC: idle ", id="status_exec")
                yield Static(" OUTPUT: idle ", id="status_output")

            yield RichLog(highlight=True, markup=True, id="plan_panel", wrap=True)
            yield RichLog(markup=True, id="metrics_panel", wrap=True)
            yield TextArea(id="output_panel", read_only=True)

        yield Input(placeholder="Run a task (example: list files)", id="input")

        yield Footer()

    # ------------------------------

    def on_mount(self):

        # status widgets
        self.plan_status = self.query_one("#status_plan", Static)
        self.exec_status = self.query_one("#status_exec", Static)
        self.output_status = self.query_one("#status_output", Static)

        # panels
        self.plan_panel = self.query_one("#plan_panel", RichLog)
        self.metrics_panel = self.query_one("#metrics_panel", RichLog)
        self.output_panel = self.query_one("#output_panel", TextArea)

        # initial text
        self.plan_panel.write("[dim]Waiting for agent plan...[/dim]")
        self.metrics_panel.write("[dim]Metrics will appear here[/dim]")
        self.output_panel.insert("[dim]Command output will appear here[/dim]")

    # ------------------------------

    def execute_agent(self, task: str):

        start = perf_counter()

        self.call_from_thread(self.plan_status.update, "⚙ Planning")

        self.call_from_thread(self.plan_panel.write, "[yellow]Generating execution plan...[/yellow]")

        plan_text, result = run_agent(task)

        self.call_from_thread(self.plan_status.update, "✓ Planning done")
        self.call_from_thread(self.plan_panel.write, plan_text)

        self.call_from_thread(self.exec_status.update, "⚙ Executing")

        command = result.get("command")
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        rc = result.get("returncode")

        self.call_from_thread(self.exec_status.update, "✓ Executed")

        duration = perf_counter() - start

        metrics = (
            f"[bold cyan]Task:[/bold cyan] {task}\n"
            f"[bold cyan]Return code:[/bold cyan] {rc}\n"
            f"[bold cyan]Execution time:[/bold cyan] {duration:.2f}s"
        )

        self.call_from_thread(self.metrics_panel.clear)
        self.call_from_thread(self.metrics_panel.write, metrics)

        self.call_from_thread(self.output_status.update, "⚙ Rendering")

        self.call_from_thread(self.output_panel.insert, f"$ {command}\n\n")

        if stdout:
            self.call_from_thread(self.output_panel.insert, stdout)

        if stderr:
            self.call_from_thread(self.output_panel.insert, "\nSTDERR:\n")
            self.call_from_thread(self.output_panel.insert, stderr)

        self.call_from_thread(self.output_status.update, "✓ Output ready")

    # ------------------------------

    def on_input_submitted(self, event: Input.Submitted):

        task = event.value.strip()
        event.input.value = ""

        if not task:
            return

        # reset UI
        self.plan_status.update("⏳ Planning")
        self.exec_status.update("⏳ Execute")
        self.output_status.update("⏳ Output")

        self.plan_panel.clear()
        self.metrics_panel.clear()
        self.output_panel.text = ""

        self.run_worker(lambda: self.execute_agent(task), thread=True)

    def action_copy_output(self):
        text = self.output_panel.text
        self.app.clipboard = text


def run():
    TerminalAgentApp().run()


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    run()
# Give me all python py files that are over 10KB in the current directory tree
