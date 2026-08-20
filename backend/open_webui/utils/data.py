import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


def convert_metadata_to_string(metadata: Dict[str, Any]) -> str:
    """Convert metadata dictionary to formatted string."""
    return "\n\n".join([f"**{key}**: {value}" for key, value in metadata.items()])


def has_fields(item: Dict[str, Any]) -> bool:
    """Check if item has 'active' and 'duration' fields."""
    return item is not None and "active" in item and "duration" in item


def array_to_markdown_table(title: str, data: List[Dict[str, Any]]) -> str:
    """Convert array of dictionaries to markdown table format."""
    # Check for invalid or empty data
    if not data or not isinstance(data, list) or len(data) == 0:
        return "No data available"
    
    # Add title as a Markdown heading
    markdown = f"# {title}\n\n"
    
    # Generate table headers
    headers = list(data[0].keys())
    markdown += "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Generate table rows
    for item in data:
        row = []
        for header in headers:
            value = item.get(header, "")
            # Escape pipe characters
            row.append(str(value).replace("|", "\\|"))
        markdown += "| " + " | ".join(row) + " |\n"
    
    return markdown


def summarize_data(data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Analyze and summarize data structure and statistics."""
    if not isinstance(data, list) or len(data) == 0:
        return None
    
    first_item = data[0]
    summary = {
        "fields": {},
        "rowCount": len(data),
        "sampleData": data[:3]  # Small sample for context
    }
    
    for key in first_item.keys():
        values = [item[key] for item in data if item.get(key) is not None]
        unique_values = list(set(values))
        
        # Determine type
        value_type = type(values[0]).__name__ if values else "string"
        if value_type == "str" and all(str(v).replace('.', '', 1).isdigit() for v in values if v):
            value_type = "number"
        
        field_info = {
            "type": value_type,
            "uniqueCount": len(unique_values),
            "min": None,
            "max": None,
            "sampleValues": unique_values[:3]
        }
        
        if value_type == "number":
            numeric_values = [float(v) for v in values if str(v).replace('.', '', 1).isdigit()]
            if numeric_values:
                field_info["min"] = min(numeric_values)
                field_info["max"] = max(numeric_values)
        
        summary["fields"][key] = field_info
    
    return summary


def generate_echarts_config(
    data: List[Dict[str, Any]],
    chart_type: str,
    title: Optional[str] = None,
    label_key: Optional[str] = None,
    value_keys: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    series_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate ECharts configuration from data."""
    
    # Validate inputs
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Data must be a non-empty array")
    
    valid_chart_types = ["bar", "line", "pie"]
    if chart_type not in valid_chart_types:
        raise ValueError(f"Invalid chart type. Must be one of: {', '.join(valid_chart_types)}")
    
    if value_keys is None:
        value_keys = []
    
    if not isinstance(value_keys, list) or len(value_keys) == 0:
        raise ValueError("value_keys must be a non-empty array of keys")
    
    if series_types and len(series_types) != len(value_keys):
        raise ValueError("series_types must match value_keys length")
    
    first_item = data[0]
    
    # Determine data types
    data_types = {}
    for key in first_item.keys():
        valid_values = [item[key] for item in data if item.get(key) is not None]
        if valid_values:
            value = valid_values[0]
            if key == "record_time":
                data_types[key] = "string"  # Force record_time to be a string
            elif isinstance(value, datetime):
                data_types[key] = "string"
            else:
                is_numeric = (isinstance(value, (int, float)) or 
                            (isinstance(value, str) and value.replace('.', '', 1).isdigit()))
                data_types[key] = "number" if is_numeric else "string"
        else:
            data_types[key] = "string"
    
    # Determine keys
    final_label_key = (label_key or 
                      next((key for key, dtype in data_types.items() if dtype == "string"), None) or 
                      "label")
    
    final_value_keys = (value_keys if value_keys else 
                       [next((key for key, dtype in data_types.items() if dtype == "number"), "value")])
    
    final_series_types = (series_types if series_types and len(series_types) == len(value_keys) else 
                         ["pie" if chart_type == "pie" else chart_type] * len(final_value_keys))
    
    # Clean and sort data
    cleaned_data = []
    for item in data:
        new_item = item.copy()
        for key, value in new_item.items():
            if value is None:
                new_item[key] = 0 if data_types[key] == "number" else "undefined"
            elif data_types[key] == "number" and isinstance(value, str):
                try:
                    new_item[key] = float(value)
                except ValueError:
                    new_item[key] = 0
            elif key == "record_time" and isinstance(value, str) and "T" in value:
                # Ensure record_time stays as ISO string
                new_item[key] = datetime.fromisoformat(value.replace('Z', '+00:00')).isoformat()
        cleaned_data.append(new_item)
    
    # Sort by time if applicable
    if (final_label_key and "time" in final_label_key and 
        cleaned_data and isinstance(cleaned_data[0].get(final_label_key), str) and 
        "T" in cleaned_data[0].get(final_label_key, "")):
        cleaned_data.sort(key=lambda x: datetime.fromisoformat(x[final_label_key].replace('Z', '+00:00')))
    
    # Validate keys exist
    for item in cleaned_data:
        if final_label_key not in item or not all(key in item for key in final_value_keys):
            raise ValueError(f"All data objects must contain {final_label_key} and {', '.join(final_value_keys)}")
    
    # Colors
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    
    # Title
    chart_title = (title or 
                  f"Data Visualization: {' and '.join(key.replace('_', ' ').upper() for key in final_value_keys)} "
                  f"by {final_label_key.replace('_', ' ').upper()}")
    
    # Prepare data
    if group_by and chart_type != "pie":
        groups = list(set(item[group_by] for item in cleaned_data if item.get(group_by) is not None))
        labels = list(set(item[final_label_key] for item in cleaned_data))
        series_data = []
        
        for group_index, group in enumerate(groups):
            for value_index, value_key in enumerate(final_value_keys):
                group_data = []
                for label in labels:
                    item = next((i for i in cleaned_data 
                               if i[final_label_key] == label and i[group_by] == group), None)
                    if item:
                        extra_props = {k: v for k, v in item.items() 
                                     if k != final_label_key and k not in final_value_keys and k != group_by}
                        group_data.append({"value": item[value_key], "extraProps": extra_props})
                    else:
                        group_data.append({"value": 0, "extraProps": {}})
                
                series_data.append({
                    "type": final_series_types[value_index],
                    "name": f"{value_key} ({group})",
                    "data": group_data,
                    "itemStyle": {
                        "color": colors[(group_index * len(final_value_keys) + value_index) % len(colors)]
                    },
                    "yAxisIndex": value_index % 2
                })
    else:
        labels = [item[final_label_key] for item in cleaned_data]
        series_data = []
        
        for index, value_key in enumerate(final_value_keys):
            if chart_type == "pie":
                data_points = []
                for data_index, item in enumerate(cleaned_data):
                    extra_props = {k: v for k, v in item.items() 
                                 if k != final_label_key and k not in final_value_keys}
                    data_points.append({
                        "name": item[final_label_key],
                        "value": item[value_key],
                        "extraProps": extra_props,
                        "itemStyle": {
                            "color": colors[(index * len(cleaned_data) + data_index) % len(colors)]
                        }
                    })
            else:
                data_points = []
                for item in cleaned_data:
                    extra_props = {k: v for k, v in item.items() 
                                 if k != final_label_key and k not in final_value_keys}
                    data_points.append({
                        "value": item[value_key],
                        "extraProps": extra_props
                    })
            
            series_info = {
                "type": final_series_types[index],
                "name": value_key.replace("_", " "),
                "data": data_points
            }
            
            if chart_type != "pie":
                series_info["itemStyle"] = {"color": colors[index % len(colors)]}
                series_info["yAxisIndex"] = index % 2
            
            series_data.append(series_info)
    
    # Base config
    config = {
        "title": {
            "text": chart_title,
            "left": "center"
        },
        "tooltip": {
            "trigger": "item" if chart_type == "pie" else "axis"
        },
        "series": series_data
    }
    
    # Axes for bar and line charts
    if chart_type in ["bar", "line"]:
        formatted_labels = []
        for value in labels:
            if isinstance(value, str) and "T" in value:
                try:
                    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    formatted_labels.append(date.strftime("%b %d, %Y %H:%M"))
                except:
                    formatted_labels.append(str(value))
            else:
                formatted_labels.append(str(value))
        
        config["xAxis"] = {
            "type": "category",
            "data": formatted_labels,
            "axisLabel": {
                "rotate": 45 if len(labels) > 5 else 0
            }
        }
        
        config["yAxis"] = []
        for index, value_key in enumerate(final_value_keys):
            unit = ""
            if "kwh" in value_key:
                unit = "kWh"
            elif "hr" in value_key:
                unit = "hr"
            
            config["yAxis"].append({
                "type": "value",
                "name": value_key.replace("_", " "),
                "position": "left" if index % 2 == 0 else "right",
                "axisLabel": {
                    "formatter": f"{{value}} {unit}"
                }
            })
        
        if group_by or len(final_value_keys) > 1:
            config["legend"] = {
                "orient": "vertical",
                "left": "left",
                "data": [s["name"] for s in series_data]
            }
    
    # Pie chart legend
    if chart_type == "pie":
        config["legend"] = {
            "orient": "vertical",
            "left": "left",
            "data": labels
        }
    
    return config


def combine_json_tool_function(data: List[List[Dict[str, Any]]], group_by: str) -> List[Dict[str, Any]]:
    """Combine JSON arrays by grouping on a specified field."""
    try:
        # Validate inputs
        if not isinstance(data, list) or not group_by:
            raise ValueError("Invalid inputs: data must be an array and group_by must be specified")
        
        # Combine JSON arrays
        combined = {}
        
        for json_array in data:
            if isinstance(json_array, list):
                for item in json_array:
                    if item and isinstance(item, dict) and group_by in item:
                        key = item[group_by]
                        if key not in combined:
                            combined[key] = item.copy()
                        else:
                            # Merge items with same group_by value
                            combined[key].update(item)
        
        # Convert combined dict to list
        result = list(combined.values())
        print("Combined JSON result:", result)
        return result
        
    except Exception as error:
        print(f"combine_json_tool error: {error}")
        raise ValueError(f"Failed to combine JSON: {str(error)}")


def clean_and_parse_json(raw_string: str) -> Optional[Any]:
    """Clean and parse JSON string, removing code block markers."""
    # Remove code block markers and trim whitespace
    cleaned = re.sub(r'```json', '', raw_string)
    cleaned = re.sub(r'```', '', cleaned)
    cleaned = cleaned.strip()
    
    try:
        # Parse the cleaned JSON string
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as error:
        print(f"Failed to parse JSON: {error}")
        return None


def array_to_list_format(array: List[str]) -> str:
    """Convert array to formatted list with S<number>: prefix."""
    # Extract the array items and map them to the desired format
    formatted_items = []
    for index, item in enumerate(array):
        # Capitalize first letter and lowercase the rest
        formatted_item = item.capitalize()
        # Add S<number>: prefix
        formatted_items.append(f"S{index + 1}: {formatted_item}")
    
    # Join items with newlines
    return "\n".join(formatted_items)