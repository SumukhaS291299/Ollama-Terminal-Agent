import os
from configparser import ConfigParser
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import AnyMessage
from rich.ansi import AnsiDecoder
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static

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

    thinking = ThinkFirst(conf)

    messages: list[AnyMessage] = MessageBuilder().build(
        SystemMessage(
            content=(
                "You are a terminal assistant. Determine the OS and Shell "
                "from the conversation history and follow those syntax rules. "
                "Use the runCommand tool whenever execution is required."
            )
        ),
        HumanMessage(content="I'm using Windows and PowerShell."),
        HumanMessage(content=task),
    )

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


# ------------------------------
# TUI Application
# ------------------------------
class TerminalAgentApp(App):
    CSS = """
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
    """

    def compose(self) -> ComposeResult:

        yield Header()

        with Vertical():
            # STATUS BAR
            with Horizontal(id="status_bar"):
                yield Static("⏳ Planning", id="status_plan")
                yield Static("⏳ Execute", id="status_exec")
                yield Static("⏳ Output", id="status_output")

            # PLAN PANEL
            yield Static("Waiting for plan...", id="plan_panel")

            # METRICS
            yield Static("Metrics will appear here", id="metrics_panel")

            # OUTPUT PANEL
            yield Static("", id="output_panel")

        yield Input(placeholder="> Enter command", id="input")

        yield Footer()

    def on_mount(self):

        self.plan_status = self.query_one("#status_plan", Static)
        self.exec_status = self.query_one("#status_exec", Static)
        self.output_status = self.query_one("#status_output", Static)

        self.plan_panel = self.query_one("#plan_panel", Static)
        self.metrics_panel = self.query_one("#metrics_panel", Static)
        self.output_panel = self.query_one("#output_panel", Static)

        self.ansi_decoder = AnsiDecoder()

    # ------------------------------

    def execute_agent(self, task: str):

        start = perf_counter()

        # planning started
        self.call_from_thread(self.plan_status.update, "⚙ Planning...")

        plan_text, result = run_agent(task)

        # planning done
        self.call_from_thread(self.plan_status.update, "✓ Planning done")

        # show plan
        self.call_from_thread(self.plan_panel.update, plan_text)

        # execution
        self.call_from_thread(self.exec_status.update, "⚙ Executing")

        command = result.get("command")
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        rc = result.get("returncode")

        self.call_from_thread(self.exec_status.update, "✓ Executed")

        # metrics
        duration = perf_counter() - start

        metrics = f"""
Task: {task}
Return code: {rc}
Execution time: {duration:.2f}s
"""

        self.call_from_thread(self.metrics_panel.update, metrics)

        # output stage
        self.call_from_thread(self.output_status.update, "⚙ Rendering")

        output_text = f"$ {command}\n\n"

        if stdout:
            output_text += stdout

        if stderr:
            output_text += "\n" + stderr

        self.call_from_thread(self.output_panel.update, output_text)

        self.call_from_thread(self.output_status.update, "✓ Output ready")

    # ------------------------------

    def on_input_submitted(self, event: Input.Submitted):

        task = event.value.strip()

        event.input.value = ""

        # reset UI
        self.plan_status.update("⏳ Planning")
        self.exec_status.update("⏳ Execute")
        self.output_status.update("⏳ Output")

        self.plan_panel.update("Planning...")
        self.metrics_panel.update("")
        self.output_panel.update("")

        self.run_worker(lambda: self.execute_agent(task), thread=True)


def run():
    TerminalAgentApp().run()


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    run()
    # main()
# Give me all python py files that are over 10KB in the current directory tree
