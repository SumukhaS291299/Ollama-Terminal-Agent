from typing import List

from langchain_core.messages import AnyMessage
from langchain_core.messages.base import BaseMessage


class MessageBuilder:
    def build(self, *messages: AnyMessage | BaseMessage) -> List[AnyMessage]:
        return list(messages)
