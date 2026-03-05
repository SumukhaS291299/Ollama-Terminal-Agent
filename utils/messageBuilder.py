from typing import List

from langchain_core.messages import AnyMessage


class MessageBuilder:

    def build(self, *messages: AnyMessage) -> List[AnyMessage]:
        return list(messages)
