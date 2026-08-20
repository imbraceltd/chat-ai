from langchain_core.messages import BaseMessage

def is_system_message(message):
    """Check if a message is a system message."""
    if isinstance(message, BaseMessage):
        return message.type == "system"
    return message.get("role") == "system"

def is_tool_message(message):
    """Check if a message is a tool message."""
    if isinstance(message, BaseMessage):
        return message.type == "tool"
    return message.get("role") == "tool"

def custom_trim_messages(messages, max_tokens=20):
    """
    Trim messages by filtering out system messages and keeping the last max_tokens messages.
    
    Args:
        messages: List of message dictionaries
        max_tokens: Maximum number of messages to keep (default: 20)
        
    Returns:
        List of trimmed messages
    """
    # Filter out system messages since includeSystem is false
    non_system_messages = [
        message for message in messages 
        if not is_system_message(message)
    ]
    
    # Determine how many messages to keep
    total_non_system = len(non_system_messages)
    to_keep = min(total_non_system, max_tokens)
    
    # Keep the last 'to_keep' messages
    trimmed_messages = non_system_messages[-to_keep:] if to_keep > 0 else []
    
    # Return the trimmed list
    return trimmed_messages