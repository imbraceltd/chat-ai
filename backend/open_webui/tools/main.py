import os
import importlib
import json
from pathlib import Path

def create_simple_tool_list():
    """Create a simple list of tools with just tool_id and description."""
    tools_dir = Path(__file__).parent  # Gets the directory of the current file
    
    tool_list = []
    
    # Get all Python files in the directory
    py_files = [f for f in tools_dir.glob("*.py") 
               if f.is_file() and f.name != "__init__.py" and f.name != "main.py"]
    
    for file_path in py_files:
        try:
            # Use file stem as tool_id
            tool_id = file_path.stem
            
            # Import the module
            module_name = f"open_webui.tools.{file_path.stem}"
            module = importlib.import_module(module_name)
            
            # Get description from function_spec if available
            description = None
            spec = None
            if hasattr(module, "Tools"):
                tool_class = getattr(module, "Tools")
                tool_instance = tool_class()
                
                if hasattr(tool_instance, "function_spec"):
                    spec = tool_instance.function_spec
                    if isinstance(spec, dict):
                        description = spec.get("description", None)
            
            tool_list.append({
                "tool_id": tool_id,
                "description": description,
                "spec": spec
            })
        
        except Exception as e:
            tool_list.append({
                "tool_id": file_path.stem,
                "description": f"Error loading tool: {str(e)}"
            })
    
    return tool_list

# Create and print the tool list
# tools = create_simple_tool_list()
# print(json.dumps(tools, indent=2))

# # For use in other files
# if __name__ == "__main__":
#     print("Available tools:")
#     for tool in tools:
#         print(f"- {tool['tool_id']}: {tool['description']}")