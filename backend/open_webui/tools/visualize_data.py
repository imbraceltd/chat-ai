from typing import List, Union, Optional
from pydantic import BaseModel, Field

class EChartsDataInput(BaseModel):
    labels: List[str] = Field(..., description="Categories or x-axis labels")
    values: List[Union[int, float]] = Field(..., description="Data values for visualization")
    series_name: Optional[str] = Field("Data", description="Name of the data series")


class Tools:
    def __init__(self):
        pass

    # Add your custom tools using pure Python code here, make sure to add type hints and descriptions
    
        pass
    tool_id = "visualize_data"
    # Define the function specification for visualize_data_with_echarts
    function_spec = {
        "description": """ 
        Generate ECharts visualization options based on the provided data and chart type.
        Args:
            data: Data to visualize containing labels and values
            chart_type: Type of chart to create (bar, line, pie, scatter)
            title: Title of the chart
            width: Width of the chart container
            height: Height of the chart container
            
        Returns:
            JSON-compatible dictionary containing ECharts options""",
        "name": "visualize_data_with_echarts",
        "parameters": {
            "properties": {
                "chart_type": {
                    "default": "bar",
                    "type": "string"
                },
                "data": {
                    "properties": {
                        "labels": {
                            "description": "Categories or x-axis labels",
                            "items": {
                                "type": "string"
                            },
                            "type": "array"
                        },
                        "series_name": {
                            "anyOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": "Data",
                            "description": "Name of the data series"
                        },
                        "values": {
                            "description": "Data values for visualization",
                            "items": {
                                "anyOf": [
                                    {
                                        "type": "integer"
                                    },
                                    {
                                        "type": "number"
                                    }
                                ]
                            },
                            "type": "array"
                        }
                    },
                    "required": [
                        "labels",
                        "values"
                    ],
                    "type": "object"
                },
                "height": {
                    "default": "500px",
                    "type": "string"
                },
                "title": {
                    "default": "Data Visualization",
                    "type": "string"
                },
                "width": {
                    "default": "800px",
                    "type": "string"
                }
            },
            "required": [
                "data"
            ],
            "type": "object"
        }
    }
    

    
    def visualize_data_with_echarts(self, data: EChartsDataInput, chart_type: str = "bar", 
                           title: str = "Data Visualization", 
                           width: str = "800px", 
                           height: str = "500px") -> dict:
        """
        Generate ECharts visualization options based on the provided data and chart type.
        
        Args:
            data: Data to visualize containing labels and values
            chart_type: Type of chart to create (bar, line, pie, scatter)
            title: Title of the chart
            width: Width of the chart container
            height: Height of the chart container
            
        Returns:
            JSON-compatible dictionary containing ECharts options
        """
        # Convert dict to EChartsDataInput if needed
        if isinstance(data, dict):
            try:
                data = EChartsDataInput(**data)
            except Exception as e:
                return {"error": f"Invalid data format: {str(e)}"}
        
        # Validate chart type
        supported_charts = ["bar", "line", "pie", "scatter", "candlestick"]
        if chart_type not in supported_charts:
            return {"error": f"Chart type '{chart_type}' not supported. Choose from {', '.join(supported_charts)}."}
        
        # Create chart options based on chart type
        if chart_type == "pie":
            # Format data for pie chart
            series_data = [{"name": label, "value": value} for label, value in zip(data.labels, data.values)]
            chart_options = {
                "title": {"text": title},
                "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b}: {c} ({d}%)"},
                "legend": {"data": data.labels, "bottom": 0},
                "series": [{
                    "name": data.series_name,
                    "type": "pie",
                    "radius": "55%",
                    "data": series_data
                }]
            }
        else:
            # Format data for bar, line, or scatter charts
            chart_options = {
                "title": {"text": title},
                "tooltip": {"trigger": "axis"},
                "legend": {"data": [data.series_name], "bottom": 0},
                "grid": {
                    "left": "5%",
                    "right": "5%",
                    "bottom": "15%",
                    "top": "15%",
                    "containLabel": True
                },
                "xAxis": {
                    "type": "category", 
                    "data": data.labels, 
                    "axisLabel": {
                        "interval": 0,
                        "rotate": 0,
                        "textStyle": {
                            "fontSize": 12
                        },
                        "margin": 15,
                        "width": 80,
                        "overflow": "break"
                    },
                    "axisTick": {
                        "alignWithLabel": True
                    }
                },
                "yAxis": {"type": "value"},
                "series": [{
                    "name": data.series_name,
                    "type": chart_type,
                    "data": data.values,
                    "barGap": "20%",  
                    "barCategoryGap": "30%",  
                    "itemStyle": {
                        "borderRadius": [4, 4, 0, 0] if chart_type == "bar" else None
                    }
                }]
            }
        
        # Return options with display configuration
        return {
            "options": chart_options,
            "config": {
                "width": width,
                "height": height
            }
        }