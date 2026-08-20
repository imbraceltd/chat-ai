from typing import List, Dict, Any
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage


def validate_tool_calls(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Validate tool calls in LangChain messages.
    Ensure AIMessage with tool_calls has a matching ToolMessage with tool_call_id.

    Args:
        messages: List of LangChain message instances.

    Returns:
        List of filtered/validated messages.
    """
    # Remove first message if it's a ToolMessage
    start_index = 0
    if messages and isinstance(messages[0], ToolMessage):
        print(f"Warning: First ToolMessage removed: {messages[0]}")
        start_index = 1

    validated_messages = []
    i = start_index

    while i < len(messages):
        msg = messages[i]

        # Handle AIMessage with tool_calls
        if (isinstance(msg, AIMessage) and 
            hasattr(msg, 'tool_calls') and 
            msg.tool_calls and 
            len(msg.tool_calls) > 0):
            
            expected = len(msg.tool_calls)
            expected_ids = {tc['id'] for tc in msg.tool_calls}

            # Check if there are enough messages left for all ToolMessages
            if i + expected >= len(messages):
                print(f"Warning: Skipping AIMessage with missing ToolMessages at end: {msg}")
                i += 1
                continue

            all_found = True
            j = i + 1

            # Check each subsequent message for matching ToolMessages
            while j < len(messages) and expected_ids:
                next_msg = messages[j]
                if (isinstance(next_msg, ToolMessage) and 
                    hasattr(next_msg, 'tool_call_id') and 
                    next_msg.tool_call_id in expected_ids):
                    expected_ids.remove(next_msg.tool_call_id)
                else:
                    all_found = False
                    break
                j += 1

            # If all tool_calls have corresponding ToolMessages
            if all_found and not expected_ids:
                validated_messages.append(msg)
                for k in range(i + 1, j):
                    validated_messages.append(messages[k])
                i = j  # Move past all processed ToolMessages
            else:
                print(f"Warning: Skipping AIMessage with incomplete ToolMessages: {msg}")
                i += 1  # Skip just the AIMessage
        
        # Handle other message types
        elif isinstance(msg, ToolMessage):
            # Skip orphaned ToolMessages
            print(f"Warning: Skipping orphaned ToolMessage: {msg}")
            i += 1
        else:
            # Include all other messages (e.g., HumanMessage)
            validated_messages.append(msg)
            i += 1

    return validated_messages


def validate_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate tool definitions.

    Args:
        tools: List of tool definition dictionaries.

    Returns:
        List of validation results with errors for each tool.
    """
    results = []
    
    for index, tool in enumerate(tools):
        errors = []

        # Check if type is "function"
        if not tool.get('type') or tool['type'] != 'function':
            errors.append("Missing or invalid 'type' (should be 'function').")

        # Check if function object exists
        function = tool.get('function')
        if not function or not isinstance(function, dict):
            errors.append("Missing or invalid 'function' object.")
        else:
            name = function.get('name')
            description = function.get('description')
            parameters = function.get('parameters')

            # Validate function name
            if not name or not isinstance(name, str):
                errors.append("Missing or invalid 'name' (should be a string).")

            # Validate function description
            if not description or not isinstance(description, str):
                errors.append("Missing or invalid 'description' (should be a string).")

            # Validate function parameters
            if not parameters or not isinstance(parameters, dict):
                errors.append("Missing or invalid 'parameters' (should be an object).")
            else:
                if parameters.get('type') != 'object':
                    errors.append("Invalid 'parameters.type' (should be 'object').")
                
                properties = parameters.get('properties')
                if not properties or not isinstance(properties, dict):
                    errors.append(
                        "Missing or invalid 'parameters.properties' (should be an object)."
                    )
                
                required = parameters.get('required')
                if required is not None and not isinstance(required, list):
                    errors.append("Invalid 'parameters.required' (should be an array).")

        result = {
            'toolIndex': index,
            'toolName': function.get('name') if function else None,
            'errors': errors
        }
        results.append(result)

    return results