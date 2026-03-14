from enum import Enum


class API(Enum):
    GENERATE = "/api/generate"
    CHAT = "/api/chat"
    EMBEDDINGS = "/api/embed"
    TAGS = "/api/tags"


class SHELL_TYPE(Enum):
    PWSH = ["pwsh", "-Command"]
    POWERSHELL = ["powershell", "-Command"]
    CMD = ["cmd", "/c"]
    BASH = ["bash", "-c"]
    SH = ["sh", "-c"]
