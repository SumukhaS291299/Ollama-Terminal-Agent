from enum import Enum


class API(Enum):
    GENERATE = "/api/generate"
    CHAT = "/api/chat"
    EMBEDDINGS = "/api/embed"
    TAGS = "/api/tags"
