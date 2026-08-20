from typing import Annotated, Literal, TypeGuard, cast
import logging
import uuid
from typing_extensions import TypedDict
from langchain_core.tools import tool, InjectedToolCallId, BaseTool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState, END
from langgraph.types import Command, Send
from langchain_core.messages import convert_to_messages
from langgraph.managed.is_last_step import RemainingSteps

logger = logging.getLogger(__name__)
METADATA_KEY_HANDOFF_DESTINATION = "__handoff_destination"
METADATA_KEY_IS_HANDOFF_BACK = "__is_handoff_back"

class State(MessagesState):
    remaining_steps: RemainingSteps
    value: str


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool

def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None, add_handoff_messages: bool = True
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool



def route_tools(state: State):
    remaining_steps = state["remaining_steps"]
    value = state["value"]
    logger.info(f"Routing tools with value: {value}")
    logger.info(f"Routing tools with remaining steps: {remaining_steps}")
    if remaining_steps <= 2:
        return END
    if value == "end":
        logger.info("Ending the graph as value is 'end'")
        return END 
    return "continue"
    

def create_node(supervisor_name: str, agent):
    def node(state: State) -> Command:
        return Command(
            update={
                "messages": []
            },
            goto=supervisor_name,
        )
    return node


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

def _has_multiple_content_blocks(content: str | list[str | dict]) -> TypeGuard[list[dict]]:
    """Check if content contains multiple content blocks."""
    return isinstance(content, list) and len(content) > 1 and isinstance(content[0], dict)

def _remove_non_handoff_tool_calls(
    last_ai_message: AIMessage, handoff_tool_call_id: str
) -> AIMessage:
    """Remove tool calls that are not meant for the agent."""
    # if the supervisor is calling multiple agents/tools in parallel,
    # we need to remove tool calls that are not meant for this agent
    # to ensure that the resulting message history is valid
    content = last_ai_message.content
    logger.info(f"Content before filtering: {last_ai_message}")
    if _has_multiple_content_blocks(content):
        content = [
            content_block
            for content_block in content
            if (content_block["type"] == "tool_use" and content_block["id"] == handoff_tool_call_id)
            or content_block["type"] != "tool_use"
        ]

    last_ai_message = AIMessage(
        content=content,
        tool_calls=[
            tool_call
            for tool_call in last_ai_message.tool_calls
            if tool_call["id"] == handoff_tool_call_id
        ],
        name=last_ai_message.name,
        id=str(uuid.uuid4()),
    )
    return last_ai_message

def create_custom_handoff_tool(
    *,
    agent_name: str,
    name: str | None = None,
    description: str | None = None,
    add_handoff_messages: bool = True,
) -> BaseTool:
    """Create a tool that can handoff control to the requested agent.

    Args:
        agent_name: The name of the agent to handoff control to, i.e.
            the name of the agent node in the multi-agent graph.
            Agent names should be simple, clear and unique, preferably in snake_case,
            although you are only limited to the names accepted by LangGraph
            nodes as well as the tool names accepted by LLM providers
            (the tool name will look like this: `transfer_to_<agent_name>`).
        name: Optional name of the tool to use for the handoff.
            If not provided, the tool name will be `transfer_to_<agent_name>`.
        description: Optional description for the handoff tool.
            If not provided, the description will be `Ask agent <agent_name> for help`.
        add_handoff_messages: Whether to add handoff messages to the message history.
            If False, the handoff messages will be omitted from the message history.
    """
    if name is None:
        name = f"transfer_to_{agent_name}"

    if description is None:
        description = f"Ask agent '{agent_name}' for help"

    @tool(name, description=description)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
            response_metadata={METADATA_KEY_HANDOFF_DESTINATION: agent_name},
        )
        last_ai_message = cast(AIMessage, state["messages"][-1])
        # Handle parallel handoffs
        # logger.info(f"Last AI message before filtering: {state}")
        if len(last_ai_message.tool_calls) > 1:
            handoff_messages = state["messages"][:-1]
            if add_handoff_messages:
                logger.info("parallel handloff")
                handoff_messages.extend(
                    (
                        _remove_non_handoff_tool_calls(last_ai_message, tool_call_id),
                        tool_message,
                    )
            )
            return Command(
                graph=Command.PARENT,
                # NOTE: we are using Send here to allow the ToolNode in langgraph.prebuilt
                # to handle parallel handoffs by combining all Send commands into a single command
                goto=[Send(agent_name, {**state, "messages": handoff_messages})],
            )
        # Handle single handoff
        else:
            if add_handoff_messages:
                handoff_messages = state["messages"] + [tool_message]
            else:
                handoff_messages = state["messages"][:-1]
            return Command(
                goto=agent_name,
                graph=Command.PARENT,
                update={**state, "messages": handoff_messages},
            )

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent


@tool
def update_tool(result: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Use this to update user information to better assist them with their questions."""
    logger.info(f"Updating tool with ID: {tool_call_id} and result: {result}")
    return Command(
        update={
            # update the state keys
     
            # update the message history
            "messages": [ToolMessage(f"Tool {tool_call_id} execute successfully", tool_call_id=tool_call_id)]
        }
    )