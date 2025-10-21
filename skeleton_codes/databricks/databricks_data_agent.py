import re
import uuid  # Import uuid
import mlflow
from mlflow.entities import SpanType
from databricks_langchain.chat_models import ChatDatabricks
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from databricks.sdk import WorkspaceClient
from databricks_langchain import DatabricksEmbeddings

# from langchain_community.vectorstores import DatabricksVectorSearch
# from databricks.vector_search.client import VectorSearchClient
from databricks import sql
from mlflow.entities import Document
from pyspark.sql import SparkSession  # Import SparkSession
from typing import (
    Any,
    Union,
    Dict,
    List,
    Optional,
    Generator,
    Optional,
    Sequence,
    Union,
)

from dataclasses import asdict
from mlflow.types.llm import ChatMessage, ChatCompletionResponse, ChatChoice

from langgraph.graph import StateGraph
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

import pandas as pd


llm_client = ChatDatabricks(endpoint="databricks-llama-4-maverick", temperature=0)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    text_retrieved_from_vector_store: Any
    sql_result: Any
    sql_guidelines: Optional[str]


def basic_node(state):
    messages = state["messages"]
    response = llm_client.invoke(messages)
    return {"messages": messages + [response]}


def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("basic_node", basic_node)
    graph.add_edge(START, "basic_node")
    graph.add_edge("basic_node", END)
    return graph.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent):
        self.agent = agent

    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]) -> list[dict]:
        """Converts a list of ChatAgentMessage (dict-like) to plain dicts for LangGraph input."""
        # ChatAgentMessage is essentially a TypedDict, so direct dict conversion should work.
        # Filter out None values if necessary, similar to the original attempt.
        return [
            {k: v for k, v in dict(msg).items() if v is not None} for msg in messages
        ]

    def _convert_langchain_message_to_chat_agent_message(self, msg) -> ChatAgentMessage:
        """Convert LangChain message objects to ChatAgentMessage format."""
        if isinstance(msg, HumanMessage):
            return ChatAgentMessage(
                role="user", content=msg.content, id=str(uuid.uuid4())
            )
        elif isinstance(msg, AIMessage):
            return ChatAgentMessage(
                role="assistant", content=msg.content, id=str(uuid.uuid4())
            )
        else:
            # Handle other message types or fallback
            return ChatAgentMessage(
                role="assistant", content=str(msg.content), id=str(uuid.uuid4())
            )

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """Non-streaming predict: process the entire conversation and return final messages."""
        # Prepare request in the format the LangGraph agent expects
        request = {"messages": self._convert_messages_to_dict(messages)}
        all_messages = []

        # Stream through the LangGraph agent events (tool calls, assistant replies, etc.)
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                # Collect all messages from each event
                for msg in node_data.get("messages", []):
                    # Convert LangChain message to ChatAgentMessage
                    converted_msg = (
                        self._convert_langchain_message_to_chat_agent_message(msg)
                    )
                    all_messages.append(converted_msg)

        # Return all accumulated messages as the response (including final answer and any tool-use messages)
        return ChatAgentResponse(messages=all_messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Streaming predict: yield chunks of the response as they become available."""
        request = {"messages": self._convert_messages_to_dict(messages)}

        # Keep track of message IDs for streaming chunks
        message_ids = {}

        # Iterate through streaming events from the LangGraph agent
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                # Yield a ChatAgentChunk for each new message in the event
                for msg in node_data.get("messages", []):
                    # Generate a consistent ID for this message content
                    # Use content hash or create new ID if not seen before
                    msg_key = f"{msg.content}_{type(msg).__name__}"
                    if msg_key not in message_ids:
                        message_ids[msg_key] = str(uuid.uuid4())

                    # Convert LangChain message to ChatAgentMessage with consistent ID
                    if isinstance(msg, HumanMessage):
                        converted_msg = ChatAgentMessage(
                            role="user", content=msg.content, id=message_ids[msg_key]
                        )
                    elif isinstance(msg, AIMessage):
                        converted_msg = ChatAgentMessage(
                            role="assistant",
                            content=msg.content,
                            id=message_ids[msg_key],
                        )
                    else:
                        converted_msg = ChatAgentMessage(
                            role="assistant",
                            content=str(msg.content),
                            id=message_ids[msg_key],
                        )

                    # Create the chunk with the properly converted message
                    yield ChatAgentChunk(delta=converted_msg)


agent = create_agent()
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
