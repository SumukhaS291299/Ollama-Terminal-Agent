# Ollama Terminal Agent

A **Textual-based terminal AI agent** that uses **Ollama models** to understand tasks and execute shell commands safely.

The agent:

- Plans the task using an LLM
- Executes commands using a tool interface
- Displays results in a structured TUI (Terminal UI)

---

## Features

- Interactive **Textual TUI**
- Works with **local Ollama models**
- Tool-based command execution
- Command output displayed with **real terminal formatting**
- Configurable shell (PowerShell, bash, cmd, etc.)

---

## Requirements

- Python **3.10+**
- **Ollama** installed and running

Example:

```bash
ollama run llama3
```

---

## Installation

Install using **uv** directly from GitHub:

```bash
uv tool install git+https://github.com/SumukhaS291299/Ollama-Terminal-Agent.git
```

---

## Usage

Start the terminal agent:

```bash
ota
```

Then type tasks like:

```
find all python files larger than 1MB
count python files in this directory
show top 10 largest files
```

The agent will:

1. Generate a plan
2. Execute commands using tools
3. Display the results in the terminal UI

---

## Configuration

Settings are stored in `config.ini`.

Example:

```
[Ollama]
Scheme = http
Host = 127.0.0.1
Port = 11434
Verify = no
Timeout = 300
; ThinkingModel = ministral-3:3b
ThinkingModel = llama3.2:latest
; ThinkingModel = ministral-3:3b

;ThinkingModel = granite4:3b
; ToolModel = ministral-3:3b
; ToolModel = functiongemma:latest
ToolModel = llama3.2:latest
; ToolModel = granite4:3b
[Shell]
; Can be any of the following: pwsh,powershell,cmd,bash,sh
type = pwsh

```

Supported shells:

- `PWSH`
- `POWERSHELL`
- `CMD`
- `BASH`
- `SH`

---

## Example Tasks

```
list python files
count python files
show files larger than 100MB
find largest files in this directory
```

---

## Development

Clone the repository:

```bash
git clone https://github.com/SumukhaS291299/Ollama-Terminal-Agent.git
cd Ollama-Terminal-Agent
```

Install locally:

```bash
uv tool install .
```

Run:

```bash
ota
```
