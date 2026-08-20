import json
import logging
import os
from typing import Dict, Any, Optional, Union, List
from langchain_openai import ChatOpenAI
from open_webui.config import OPENAI_API_KEY, OPENAI_API_BASE_URLS
from open_webui.internal.mongo_db import mongodb_client

logger = logging.getLogger(__name__)


class EChartGenerator:
    """Service for generating Apache ECharts JSON configurations from user questions and data."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # Route the ECharts-generation LLM through the shared provider factory so
        # it follows the same OpenAI->Bedrock migration as the rest of the app.
        # ECHART_PROVIDER / ECHART_MODEL override the provider/model; when unset
        # we fall back to the app-wide default provider (LLM_DEFAULT_PROVIDER).
        # Lazy import to avoid a circular import with llm.agent.
        from open_webui.llm.agent import get_llm_provider, CONFIG, PROVIDER

        self.provider_name = (
            os.getenv("ECHART_PROVIDER") or PROVIDER or "openai"
        ).strip().lower()

        if self.provider_name == "openai":
            # Legacy OpenAI path (also honours an explicit api_key/base_url and the
            # OpenAI proxy in OPENAI_API_BASE_URLS).
            self.api_key = api_key or OPENAI_API_KEY
            self.base_url = base_url or (
                OPENAI_API_BASE_URLS.value[0] if OPENAI_API_BASE_URLS.value else None
            )
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            self.llm = ChatOpenAI(
                model=os.getenv("ECHART_MODEL", "gpt-4.1"),
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.2,
            )
        else:
            # Bedrock / other providers: build the LLM via the provider factory.
            # available_models is unused by non-OpenAI create_llm, so pass {}.
            # Empty ECHART_MODEL -> the provider's own default model.
            provider = get_llm_provider(self.provider_name, CONFIG)
            self.llm = provider.create_llm(
                os.getenv("ECHART_MODEL", ""), {}, streaming=False, temperature=0.2
            )

    async def generate_echart(
        self, question: str, data: Optional[str] = None
    ) -> Union[Dict[str, Any], str]:
        """
        Generate Apache ECharts JSON configuration from user question and data.

        Args:
            question: User's question about the data
            data: File content or data to analyze (optional)

        Returns:
            Apache ECharts JSON configuration or "no echart required" if not applicable
        """
        try:
            # Create the prompt for the LLM
            system_prompt = """You are an expert data visualization assistant. Your task is to generate Apache ECharts JSON configurations from user questions and data.

CRITICAL RULE: When data is provided, you MUST ALWAYS generate a valid Apache ECharts JSON configuration. NEVER respond with "no echart required" when data is available. This is a dedicated chart generation endpoint - if data is provided, a chart is ALWAYS expected.

You should generate a chart for ALL types of questions when data is present, including:
- Factual questions about specific values → Generate a chart that visualizes ALL the data with emphasis on the relevant item
- Comparison questions → Generate charts comparing the data points
- Trend/pattern questions → Generate appropriate trend visualizations
- Display/visualization requests → Generate the requested visualization
- Any other question → Pick the best chart type based on the data structure

IMPORTANT - Handling specific-value questions:
- When the user asks about a SPECIFIC item in the data, STILL visualize ALL the data
- Use visual emphasis (e.g., different color, highlight) on the specific item the user asked about
- Generate a SHORT, concise chart title (maximum 6 words). Do NOT use the full question as the title. Example: "Microsoft 2025 Financials" instead of "What are Microsoft's 2025 Income Statement metrics including Total Revenue, Net Income, and Operating Cash Flow?"

CRITICAL - Smart Chart Type Selection:
- If the user requests a specific chart type (calendar, timeline, etc.) but the data doesn't have the required fields (e.g., dates), AUTOMATICALLY select the MOST APPROPRIATE alternative chart type
- NEVER respond with "I can't create X" or "no echart required" just because the requested format doesn't match
- ALWAYS generate a chart by choosing the best alternative based on the data structure
- For categorical data with counts/values: use bar charts, pie charts, or horizontal bar charts
- For comparisons: use bar charts or grouped bar charts
- For data with long text descriptions: use table charts
- For hierarchical data: use treemap or sunburst charts
- For relationships: use network graphs or chord diagrams

Data Structure Guidelines:
- Data with party/category + count/value → Bar chart or Pie chart
- Data with multiple numeric columns → Table chart or Multi-series bar/line chart
- Data with dates/time → Line chart or Calendar heatmap
- Data with long text + short values → Table chart with proper column widths
- Data with categories and subcategories → Treemap or Sunburst
- "Show all data" requests with structured data → Table chart for comprehensive display

When generating ECharts JSON:
- Use appropriate chart types (bar, line, pie, scatter, table, treemap, etc.) based on the data structure
- Ensure the JSON is valid and follows Apache ECharts format
- Include proper titles, legends, tooltips, and data formatting
- Use the provided data to populate the chart
- Make the chart informative and visually appealing
- CRITICAL: Output ONLY valid JSON - NO JavaScript functions (e.g., formatter: function, color: function)
- CRITICAL: NO comments (//) in the JSON
- CRITICAL: For conditional coloring (e.g., highlighting a specific bar), use the data array with individual itemStyle objects instead of a function. Example: "data": [{"value": 20, "itemStyle": {"color": "#87cefa"}}, {"value": 24, "itemStyle": {"color": "#ff7f50"}}]
- CRITICAL: NEVER use "function" keyword anywhere in the JSON output
- CRITICAL: For axis labels, DO NOT include formatter properties - format the data correctly before adding it to the chart
- CRITICAL: If data labels/names are very long (>50 characters), create SHORT, CLEAR labels for the chart
- CRITICAL: For long labels, use abbreviated/summarized versions (10-30 chars) in the name field
- CRITICAL: If you need custom formatting, format the data values directly in the data array, not with formatters
- Simple template strings like "{value}", "{a}", "{b}", "{c}" are allowed in tooltips only
- CRITICAL: ECharts tooltip formatter does NOT support array index syntax like "{c[0]}" or "{c[1]}". To reference a specific series by index, use "{c0}", "{c1}", "{a0}", "{a1}", "{b0}", "{b1}" (digit directly after the letter, NO brackets)
- CRITICAL: For multi-series tooltips with axis trigger, PREFER omitting the formatter entirely — ECharts will auto-generate a correct multi-series tooltip. Only set a formatter if you truly need custom formatting, and then use "{a0}: {c0}<br/>{a1}: {c1}" style
- CRITICAL - Multi-series with different magnitudes: When a chart has 2+ bar/line series whose value ranges differ by 10x or more (e.g. counts ~50 vs currency ~2,000,000), you MUST use dual Y-axes. Set yAxis to an array of two axis objects (left for the smaller-scale series, right for the larger-scale series), and set "yAxisIndex": 0 / "yAxisIndex": 1 on the corresponding series. Otherwise the smaller series becomes invisible because the axis scales to the larger one.
- CRITICAL: For Sankey diagrams, ensure data is a Directed Acyclic Graph (DAG) with NO CYCLES
- CRITICAL: Never use Sankey diagrams if the data has circular relationships or bidirectional flows
- CRITICAL: For cyclic or bidirectional data, use chord diagrams, network graphs, or bar/line charts instead
- CRITICAL: In Sankey diagrams, ALL node names must be UNIQUE - prefix outcomes with their category (e.g., "IP: Standard", "Subcontracting: Standard")
- CRITICAL: If multiple categories have the same outcome names, make them unique by prepending the category name

Only respond with "no echart required" if NO data is provided AND the question is purely conversational (e.g., "Hello", "How are you?").

Data format: {data}

Question: {question}

Analyze the data structure and select the most appropriate chart type. If the user's requested format doesn't match the data (e.g., calendar format without dates), automatically choose the best alternative visualization. Respond with a valid Apache ECharts JSON object. The title.text MUST be short and concise (maximum 6 words) — do NOT use the full question as the title.
"""

            user_prompt = f"Data: {data or 'No data provided'}\n\nQuestion: {question}"

            # Make the request to OpenAI
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.llm.ainvoke(messages)

            if not response or not hasattr(response, "content"):
                logger.error("No response received from LLM")
                return "no echart required"

            # Normalise content: some providers (e.g. Bedrock Converse) return a
            # list of content blocks instead of a plain string.
            raw_content = response.content
            if isinstance(raw_content, list):
                raw_content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_content
                )
            content = (raw_content or "").strip()

            # Check if response is "no echart required"
            if content.lower() == "no echart required":
                return "no echart required"

            # Try to parse as JSON, handling potential markdown code blocks
            try:
                # Clean the response content
                cleaned_content = content.strip()

                # Remove markdown code block wrappers if present
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]  # Remove ```json
                if cleaned_content.startswith("```"):
                    cleaned_content = cleaned_content[3:]  # Remove ```
                if cleaned_content.endswith("```"):
                    # Remove trailing ```
                    cleaned_content = cleaned_content[:-3]

                # Clean up any remaining whitespace
                cleaned_content = cleaned_content.strip()

                import re
                # Remove single-line comments: // comment
                cleaned_content = re.sub(r'//.*', '', cleaned_content)
                # Remove trailing commas before closing braces/brackets
                cleaned_content = re.sub(
                    r',(\s*[}\]])', r'\1', cleaned_content)

                echart_config = json.loads(cleaned_content)
                # Normalize malformed title objects like {"Title Text": ""}
                try:
                    title = echart_config.get("title") if isinstance(
                        echart_config, dict) else None
                    if isinstance(title, dict) and "text" not in title and len(title) == 1:
                        # Convert {"Some Title": ""} -> {"text": "Some Title"}
                        only_key = next(iter(title.keys()))
                        if isinstance(only_key, str) and only_key.strip():
                            echart_config["title"] = {"text": only_key}
                except Exception as e:
                    logger.warning(f"Failed to normalize title field: {e}")
                # Validate that it's a dictionary (ECharts config should be an object)
                if isinstance(echart_config, dict):
                    # Fix invalid tooltip formatter syntax like "{c[0]}" / "${c[1]}"
                    # Reason: ECharts does not support array-index tokens; they
                    # render as literal text in the UI.
                    self._fix_tooltip_formatters(echart_config)
                    self._fix_mismatched_scales(echart_config)

                    # Validate and fix Sankey diagrams
                    if self._is_sankey_chart(echart_config):
                        validation = self._validate_sankey_nodes(echart_config)
                        if not validation["valid"]:
                            logger.warning(
                                f"Sankey validation failed: {validation['error']}")
                            logger.warning(
                                f"Attempting to fix duplicate nodes: {validation.get('duplicates', [])}")
                            echart_config = self._fix_sankey_duplicates(
                                echart_config)
                            # Re-validate after fix
                            validation = self._validate_sankey_nodes(
                                echart_config)
                            if not validation["valid"]:
                                logger.error(
                                    "Failed to fix Sankey diagram, rejecting chart")
                                return "no echart required"

                    # Validate and fix graph/network charts
                    if self._is_graph_chart(echart_config):
                        validation = self._validate_graph_chart(echart_config)
                        if not validation["valid"]:
                            logger.warning(
                                f"Graph validation failed: {validation['error']}")
                            if validation.get('self_loops'):
                                logger.warning(
                                    f"Removing self-loops: {validation['self_loops']}")
                            if validation.get('duplicates'):
                                logger.warning(
                                    f"Fixing duplicate nodes: {validation['duplicates']}")
                            echart_config = self._fix_graph_issues(
                                echart_config)
                            # Re-validate after fix
                            validation = self._validate_graph_chart(
                                echart_config)
                            if not validation["valid"]:
                                logger.error(
                                    "Failed to fix graph chart, rejecting chart")
                                return "no echart required"

                    logger.info("Successfully generated ECharts configuration")
                    return echart_config
                else:
                    logger.warning(
                        f"Response is not a valid ECharts object: {type(echart_config)}"
                    )
                    return "no echart required"
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response as JSON: {e}")
                logger.error(f"Raw response: {content}")
                return "no echart required"

        except Exception as e:
            logger.error(f"Error generating ECharts configuration: {e}")
            return "no echart required"

    def _fix_tooltip_formatters(self, config: Dict[str, Any]) -> None:
        """Rewrite invalid ECharts tooltip formatter tokens in-place.

        ECharts only supports {a}/{b}/{c}/{d} optionally suffixed by a digit
        (e.g. {c0}, {c1}). The LLM sometimes emits JavaScript-style array
        indexing like ${c[0]} or {c[1]}, which ECharts renders as literal
        text. Strip the leading '$' and convert [N] to N so the tokens
        become valid.
        """
        import re

        token_pattern = re.compile(r"\$?\{([abcd])\[(\d+)\]\}")

        def _fix_string(s: str) -> str:
            # Convert ${c[0]} or {c[0]} -> {c0}
            fixed = token_pattern.sub(r"{\1\2}", s)
            # Strip stray '$' immediately before a valid {x}/{xN} token
            fixed = re.sub(r"\$(\{[abcd]\d*\})", r"\1", fixed)
            return fixed

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                for k, v in list(node.items()):
                    if k == "formatter" and isinstance(v, str):
                        node[k] = _fix_string(v)
                    else:
                        _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        try:
            _walk(config)
        except Exception as e:
            logger.warning(f"Failed to fix tooltip formatters: {e}")

    def _fix_mismatched_scales(self, config: Dict[str, Any]) -> None:
        """Rewrite single-Y-axis configs into dual-Y-axis when series scales differ by 10x+.

        Why: ECharts auto-scales the shared axis to the largest series, making
        smaller-scale series (e.g. counts alongside currency) visually invisible.
        """
        try:
            series = config.get("series")
            if not isinstance(series, list) or len(series) < 2:
                return

            if not all(
                isinstance(s, dict) and s.get("type") in ("bar", "line")
                for s in series
            ):
                return

            y_axis = config.get("yAxis")
            if isinstance(y_axis, list):
                return

            scales: List[float] = []
            for s in series:
                data = s.get("data")
                if not isinstance(data, list) or not data:
                    return
                nums: List[float] = []
                for v in data:
                    if isinstance(v, (int, float)):
                        nums.append(abs(float(v)))
                    elif isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
                        nums.append(abs(float(v["value"])))
                if not nums:
                    return
                scales.append(max(nums))

            min_scale = min(scales)
            max_scale = max(scales)
            if min_scale <= 0:
                return
            ratio = max_scale / min_scale
            if ratio < 10:
                return

            sorted_scales = sorted(scales)
            mid = sorted_scales[len(sorted_scales) // 2]

            small_name = next(
                (s.get("name") for s, sc in zip(series, scales) if sc <= mid and s.get("name")),
                "Count",
            )
            large_name = next(
                (s.get("name") for s, sc in zip(series, scales) if sc > mid and s.get("name")),
                "Value",
            )

            base_axis = y_axis if isinstance(y_axis, dict) else {"type": "value"}
            left_axis = {**base_axis, "type": base_axis.get("type", "value"),
                         "name": small_name, "position": "left"}
            right_axis = {"type": base_axis.get("type", "value"),
                          "name": large_name, "position": "right"}
            config["yAxis"] = [left_axis, right_axis]

            for s, sc in zip(series, scales):
                s["yAxisIndex"] = 1 if sc > mid else 0

            tooltip = config.get("tooltip")
            if isinstance(tooltip, dict) and tooltip.get("trigger") == "axis":
                fmt = tooltip.get("formatter")
                if isinstance(fmt, str):
                    import re
                    referenced = set(re.findall(r"\{c(\d+)\}", fmt))
                    expected = {str(i) for i in range(len(series))}
                    if not expected.issubset(referenced):
                        tooltip.pop("formatter", None)

            logger.info(
                f"Fixed mismatched-scale chart: ratio={ratio:.1f}x, "
                f"series_scales={scales}, split_at={mid}"
            )
        except Exception as e:
            logger.warning(f"Failed to fix mismatched scales: {e}")

    def _is_sankey_chart(self, config: Dict[str, Any]) -> bool:
        """Check if the chart is a Sankey diagram."""
        try:
            series = config.get("series", [])
            return isinstance(series, list) and len(series) > 0 and series[0].get("type") == "sankey"
        except Exception:
            return False

    def _is_graph_chart(self, config: Dict[str, Any]) -> bool:
        """Check if the chart is a graph/network chart."""
        try:
            series = config.get("series", [])
            return isinstance(series, list) and len(series) > 0 and series[0].get("type") == "graph"
        except Exception:
            return False

    def _validate_sankey_nodes(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Sankey diagram for duplicate node names."""
        try:
            series = config.get("series", [])
            if not series:
                return {"valid": True}

            nodes = series[0].get("data", [])
            node_names = [n.get("name")
                          for n in nodes if isinstance(n, dict) and "name" in n]

            # Check for duplicates
            seen, duplicates = set(), set()
            for name in node_names:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)

            if duplicates:
                return {
                    "valid": False,
                    "error": "Duplicate node names found",
                    "duplicates": list(duplicates)
                }
            return {"valid": True}
        except Exception as e:
            logger.warning(f"Sankey validation error: {e}")
            return {"valid": True}

    def _validate_graph_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate graph chart for self-loops and duplicate nodes."""
        try:
            series = config.get("series", [])
            if not series:
                return {"valid": True}

            graph = series[0]
            nodes = graph.get("data", [])
            links = graph.get("links", [])

            # Check duplicates
            node_names = [n.get("name")
                          for n in nodes if isinstance(n, dict) and "name" in n]
            seen, duplicates = set(), set()
            for name in node_names:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)

            # Check self-loops
            self_loops = []
            for link in links:
                if isinstance(link, dict):
                    src, tgt = link.get("source"), link.get("target")
                    if src and tgt and src == tgt:
                        self_loops.append(src)

            if self_loops or duplicates:
                result = {"valid": False, "error": "Graph chart has issues"}
                if self_loops:
                    result["self_loops"] = list(set(self_loops))
                if duplicates:
                    result["duplicates"] = list(duplicates)
                return result

            return {"valid": True}
        except Exception as e:
            logger.warning(f"Graph validation error: {e}")
            return {"valid": True}

    def _fix_sankey_duplicates(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fix duplicate node names in Sankey diagram by adding suffixes."""
        try:
            series = config.get("series", [])
            if not series:
                return config

            sankey = series[0]
            nodes = sankey.get("data", [])
            links = sankey.get("links", [])

            # Track name occurrences and create mapping
            name_count = {}
            name_mapping = {}

            for node in nodes:
                if not isinstance(node, dict) or "name" not in node:
                    continue

                original_name = node["name"]
                if original_name not in name_count:
                    name_count[original_name] = 0
                    name_mapping[original_name] = []

                name_count[original_name] += 1
                if name_count[original_name] > 1:
                    new_name = f"{original_name} ({name_count[original_name]})"
                    node["name"] = new_name
                    name_mapping[original_name].append(
                        (original_name, new_name))
                else:
                    name_mapping[original_name].append(
                        (original_name, original_name))

            # Update links to use new names
            for link in links:
                if not isinstance(link, dict):
                    continue
                for old_name, mappings in name_mapping.items():
                    if len(mappings) > 1:
                        # For duplicates, try to match by position/context
                        # For simplicity, we keep the first occurrence unchanged
                        for idx, (orig, new) in enumerate(mappings):
                            if idx > 0:  # Update duplicates
                                if link.get("source") == orig:
                                    link["source"] = new
                                if link.get("target") == orig:
                                    link["target"] = new

            logger.info("Fixed Sankey duplicate nodes")
            return config
        except Exception as e:
            logger.error(f"Error fixing Sankey duplicates: {e}")
            return config

    def _fix_graph_issues(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fix graph chart issues: remove self-loops and fix duplicate nodes."""
        try:
            series = config.get("series", [])
            if not series:
                return config

            graph = series[0]
            nodes = graph.get("data", [])
            links = graph.get("links", [])

            # Fix duplicate nodes by adding category prefix
            name_count = {}
            name_mapping = {}

            for node in nodes:
                if not isinstance(node, dict) or "name" not in node:
                    continue

                original_name = node["name"]
                category = node.get("category", "")

                if original_name not in name_count:
                    name_count[original_name] = 0
                    name_mapping[original_name] = []

                name_count[original_name] += 1
                if name_count[original_name] > 1 and category:
                    # Add category prefix for duplicates
                    new_name = f"{category}: {original_name}"
                    old_name = node["name"]
                    node["name"] = new_name
                    name_mapping[original_name].append((old_name, new_name))
                else:
                    name_mapping[original_name].append(
                        (original_name, original_name))

            # Update links and remove self-loops
            filtered_links = []
            for link in links:
                if not isinstance(link, dict):
                    continue

                src, tgt = link.get("source"), link.get("target")

                # Update with new names
                for old_name, mappings in name_mapping.items():
                    for old, new in mappings:
                        if src == old:
                            link["source"] = new
                            src = new
                        if tgt == old:
                            link["target"] = new
                            tgt = new

                # Filter out self-loops
                if src != tgt:
                    filtered_links.append(link)
                else:
                    logger.info(f"Removed self-loop: {src} -> {tgt}")

            graph["links"] = filtered_links
            logger.info("Fixed graph chart issues")
            return config
        except Exception as e:
            logger.error(f"Error fixing graph issues: {e}")
            return config


# Factory function
def create_echart_service(
    api_key: Optional[str] = None, base_url: Optional[str] = None
) -> EChartGenerator:
    """Create an EChart generator service instance."""
    return EChartGenerator(api_key, base_url)


# Module-level async function for convenience
async def generate_echart(
    question: str,
    data: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Union[Dict[str, Any], str]:
    """
    Generate Apache ECharts JSON configuration from user question and data.

    Args:
        question: User's question about the data
        data: File content or data to analyze (optional)
        api_key: OpenAI API key (optional, uses config default)
        base_url: OpenAI base URL (optional, uses config default)

    Returns:
        Apache ECharts JSON configuration or "no echart required" if not applicable
    """
    service = create_echart_service(api_key, base_url)
    return await service.generate_echart(question, data)


async def get_echarts_by_thread_id(thread_id: str) -> List[Dict[str, Any]]:
    """Get all echarts for a specific thread_id, ordered by created_at descending."""
    try:
        from open_webui.repository.echart import echart_repo
        echarts = await echart_repo.get_by_thread_id(thread_id)
        logger.info(f"Retrieved {len(echarts)} echarts for thread_id: {thread_id}")
        return echarts
    except Exception as e:
        logger.error(f"Error retrieving echarts for thread_id {thread_id}: {e}")
        return []
