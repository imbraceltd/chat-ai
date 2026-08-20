from typing import Any, Dict,  BinaryIO
import os
import uuid
import tempfile
import re
import logging
import time
import gc
import pdfplumber
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel

import json

# --- Batch processing config (override via environment variables) ---
PDF_BATCH_SIZE = int(os.environ.get("PDF_BATCH_SIZE", "50"))  # pages per pdfplumber session
PDF_MAX_PAGES = int(os.environ.get("PDF_MAX_PAGES", "2000"))
PDF_MAX_FILE_SIZE_MB = int(os.environ.get("PDF_MAX_FILE_SIZE_MB", "200"))
PDF_MAX_FILE_SIZE = PDF_MAX_FILE_SIZE_MB * 1024 * 1024

log = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel

class Element(BaseModel):
    type: str
    text: Any
    page: int = 0


def generate_table_summaries_batch(
    table_elements: list,
    batch_size: int = 10,
) -> list:
    """Use the retriever LLM to generate natural language summaries for a batch
    of table elements. Each summary is prepended to the original table text to
    improve embedding quality while preserving the full table data.

    Args:
        table_elements: List of Element objects with type="table".
        batch_size: Number of tables to summarize per LLM call.

    Returns:
        The same list with each element's text updated (summary + original).
    """
    if not table_elements:
        return table_elements

    try:
        from open_webui.llm.agent import get_llm_provider, CONFIG, RETRIEVER_PROVIDER
        from langchain.schema import HumanMessage

        provider_name = RETRIEVER_PROVIDER if RETRIEVER_PROVIDER else "openai"
        provider = get_llm_provider(provider_name, CONFIG)
        llm = provider.create_retriever_model(model=None, temperature=0.0)

        for i in range(0, len(table_elements), batch_size):
            batch = table_elements[i : i + batch_size]

            # Build a single prompt with all tables in this batch
            tables_text = ""
            for idx, elem in enumerate(batch):
                tables_text += (
                    f"=== TABLE {idx + 1} (Page {elem.page}) ===\n"
                    f"{elem.text}\n\n"
                )

            prompt = (
                "For each table below, write a 1-2 sentence summary describing what data "
                "it contains. Mention the key metrics, column headers, and units if visible. "
                "Be specific and factual.\n\n"
                "Return your response as a numbered list matching each table:\n"
                "1. <summary for TABLE 1>\n"
                "2. <summary for TABLE 2>\n"
                "...\n\n"
                f"{tables_text}"
            )

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                content = response.content.strip()

                # Parse numbered summaries from response
                summaries = _parse_numbered_summaries(content, len(batch))

                for idx, elem in enumerate(batch):
                    if idx < len(summaries) and summaries[idx]:
                        elem.text = f"{summaries[idx]}\n\n{elem.text}"

            except Exception as e:
                log.warning(f"Batch table summary failed for batch {i // batch_size + 1}: {e}")

        log.info(f"Generated summaries for {len(table_elements)} tables in {(len(table_elements) - 1) // batch_size + 1} batch(es)")

    except Exception as e:
        log.warning(f"Table summary generation setup failed: {e}")

    return table_elements


def _parse_numbered_summaries(content: str, expected_count: int) -> list:
    """Parse numbered summaries from LLM response like '1. ...\n2. ...'"""
    summaries = [""] * expected_count
    lines = content.strip().split("\n")

    current_idx = -1
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a number like "1." or "1:"
        match = re.match(r"^(\d+)[.):]\s*(.*)", line)
        if match:
            # Save previous summary
            if current_idx >= 0 and current_idx < expected_count:
                summaries[current_idx] = " ".join(current_text).strip()

            current_idx = int(match.group(1)) - 1  # 1-indexed to 0-indexed
            current_text = [match.group(2)] if match.group(2) else []
        elif current_idx >= 0:
            current_text.append(line)

    # Save last summary
    if current_idx >= 0 and current_idx < expected_count:
        summaries[current_idx] = " ".join(current_text).strip()

    return summaries

_COLUMN_HEADER_KEYWORDS = frozenset([
    'name', 'type', 'value', 'amount', 'date', 'time', 'category', 'id', 'number',
    'count', 'total', 'sum', 'average', 'mean', 'min', 'max', 'year', 'month', 'day',
    'description', 'title', 'status', 'code', 'rate', 'price', 'cost', 'quantity',
    'size', 'weight', 'length', 'width', 'height', 'area', 'volume', 'percent',
    'percentage', 'ratio', 'index', 'score', 'rank', 'level', 'grade', 'class',
    'item', 'product', 'service', 'customer', 'order', 'invoice', 'payment',
])

_COMPOUND_TABLE_TITLE_INDICATORS = frozenset([
    'table', 'section', 'part', 'category', 'type', 'group',
    'summary', 'total', 'subtotal', 'breakdown', 'analysis',
    'quarter', 'year', 'month', 'period', 'phase', 'continued',
    'cont.', 'continued...', '(continued)', 'part 1', 'part 2',
])

_TITLE_KEYWORDS = frozenset([
    'INTRODUCTION', 'CONCLUSION', 'ABSTRACT', 'SUMMARY',
    'METHODOLOGY', 'RESULTS', 'DISCUSSION', 'REFERENCES',
    'BACKGROUND', 'LITERATURE', 'ANALYSIS', 'FINDINGS'
])

_NUMBERED_HEADING_RE = re.compile(r'^\d+\.?\d*\.?\s+[A-Z]')


def _detect_titles_on_page(page_text, page_num):
    """Detect title/heading lines from already-extracted page text."""
    titles = []
    lines = page_text.split('\n')
    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (
            (stripped.isupper() and len(stripped) < 100) or
            _NUMBERED_HEADING_RE.match(stripped) or
            any(kw in stripped.upper() for kw in _TITLE_KEYWORDS)
        ):
            titles.append({
                'page': page_num + 1,
                'line': line_num,
                'text': stripped,
                'type': 'heading'
            })
    return titles


def _extract_text_with_titles(page_text, page_titles, page_num, text_elements):
    """Extract text chunks using title-based sectioning."""
    lines = page_text.split('\n')
    current_section = None
    current_content = []

    # Build a set for fast title lookup
    title_lookup = {(t['line'], t['text']) for t in page_titles}

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if (line_num, line) in title_lookup:
            # Save previous section if it has content
            if current_section and current_content:
                section_text = '\n'.join(current_content)
                if len(section_text.strip()) > 20:
                    text_elements.append(Element(
                        type="text",
                        text=f"Page {page_num + 1}, Section: {current_section}\n\n{section_text}"
                    ))
            # Start new section
            current_section = line
            current_content = []
        else:
            if line and not (len(line) < 5 and line.isdigit()):
                current_content.append(line)

    # Don't forget the last section
    if current_section and current_content:
        section_text = '\n'.join(current_content)
        if len(section_text.strip()) > 20:
            text_elements.append(Element(
                type="text",
                text=f"Page {page_num + 1}, Section: {current_section}\n\n{section_text}"
            ))


def _extract_text_with_paragraphs(page_text, page_num, text_elements):
    """Extract text chunks using paragraph-based grouping."""
    lines = page_text.split('\n')
    paragraphs = []
    current_paragraph = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                paragraphs.append('\n'.join(current_paragraph))
                current_paragraph = []
        else:
            if (len(line) < 5 and line.isdigit()) or \
               (len(line) < 20 and any(word in line.lower() for word in ['page', 'chapter', 'section'])):
                continue
            current_paragraph.append(line)

    if current_paragraph:
        paragraphs.append('\n'.join(current_paragraph))

    chunk_size = 3
    for i in range(0, len(paragraphs), chunk_size):
        chunk_paragraphs = paragraphs[i:i + chunk_size]
        chunk_text = '\n\n'.join(chunk_paragraphs)

        if len(chunk_text.strip()) > 30:
            first_line = chunk_paragraphs[0].split('\n')[0] if chunk_paragraphs else "Content"
            section_hint = first_line[:50] + "..." if len(first_line) > 50 else first_line
            text_elements.append(Element(
                type="text",
                text=f"Page {page_num + 1}, Content: {section_hint}\n\n{chunk_text}"
            ))


def _extract_text_fallback(page_text, page_num, text_elements):
    """Fallback word-based chunking for pages with very short or unusual text."""
    if not page_text or len(page_text.strip()) <= 20:
        return
    words = page_text.split()
    chunk_size = 300
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        if len(chunk_text.strip()) > 30:
            context = ' '.join(chunk_words[:10]) + "..."
            text_elements.append(Element(
                type="text",
                text=f"Page {page_num + 1}, Content: {context}\n\n{chunk_text}"
            ))


_TABLE_TEXT_PATTERN = re.compile(
    r'(?:'
    r'\b(?:HK\$|US\$|RMB|EUR|RM)\s*[\d,]+(?:\.\d+)?[MBK]?\b'  # currency values
    r'|[\d,]+\.?\d*\s*%'                                         # percentages
    r'|\b\d{4}/\d{2,4}\b'                                        # fiscal year patterns
    r')',
    re.IGNORECASE,
)

_TABLE_HEADER_KEYWORDS = re.compile(
    r'\b(?:Turnover|Revenue|Sales|Profit|Loss|Total|Segment|'
    r'Year-on-year|YoY|Change|Margin|Amount|Online|Offline|'
    r'HK\s*\$\s*Million|HK\$\'000)\b',
    re.IGNORECASE,
)


def _page_has_table_indicators(page, page_text=None):
    """Quick check if a page likely contains tables by looking for lines/edges
    or text patterns that suggest tabular data (e.g. borderless/styled tables).
    This is much cheaper than extract_tables() which runs full table detection."""
    try:
        # Check for horizontal/vertical lines that indicate table structure
        lines = page.lines
        if lines and len(lines) >= 3:
            return True
        # Check for rectangle objects (table cell borders)
        rects = page.rects
        if rects and len(rects) >= 2:
            return True
        # Check for edge objects
        edges = page.edges
        if edges and len(edges) >= 4:
            return True

        # Fallback: check text content for tabular patterns (catches borderless
        # tables styled with colors/spacing instead of drawn grid lines)
        if page_text:
            text_lines = page_text.split('\n')
            # Count lines that contain multiple numeric/currency values — a strong
            # signal of tabular data even without visible borders
            data_line_count = 0
            has_header_keyword = bool(_TABLE_HEADER_KEYWORDS.search(page_text))
            for line in text_lines:
                matches = _TABLE_TEXT_PATTERN.findall(line)
                if len(matches) >= 2:
                    data_line_count += 1
            # At least 2 data-like lines + a header keyword → likely a table
            if has_header_keyword and data_line_count >= 2:
                return True

        return False
    except Exception:
        return True  # Fall back to trying extract_tables() on error


def _split_pdf_to_temp(pdf_path, start_page, end_page):
        """Split a range of pages from a PDF into a temporary file using pypdf.

        Args:
            pdf_path: Path to the source PDF.
            start_page: 0-based start index (inclusive).
            end_page: 0-based end index (exclusive).

        Returns:
            Path to the temporary PDF file containing the requested pages.
        """
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        for i in range(start_page, min(end_page, len(reader.pages))):
            writer.add_page(reader.pages[i])

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            writer.write(tmp)
        finally:
            tmp.close()
        return tmp.name


def _extract_batch(chunk_path, batch_start, total_pages, text_elements, table_elements, stats):
        """Run pdfplumber extraction on a single PDF chunk.

        This is the same per-page logic that was previously inside
        ``extract_pdf_elements``, but scoped to a small chunk so that
        pdfplumber (and pdfminer underneath) never holds the full
        document structure in memory.

        *stats* is a dict updated in-place with timing / counter data.
        """
        with pdfplumber.open(chunk_path, laparams={"boxes_flow": None}) as pdf:
            for local_idx, page in enumerate(pdf.pages):
                # Absolute page number in the original document
                page_num = batch_start + local_idx

                if (page_num + 1) % 50 == 0:
                    elapsed = time.time() - stats["start_time"]
                    log.info(f"Processing page {page_num + 1}/{total_pages} ({elapsed:.1f}s elapsed)")

                # --- Extract text once per page ---
                t0 = time.time()
                try:
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if not page_text:
                        page_text = page.extract_text_simple(x_tolerance=3, y_tolerance=3)
                except Exception as e:
                    try:
                        page_text = page.extract_text_simple(x_tolerance=3, y_tolerance=3)
                    except Exception:
                        log.error(f"Error extracting text from page {page_num + 1}: {e}")
                        page_text = None
                stats["text_time"] += time.time() - t0

                # --- Table extraction (only if page has table indicators) ---
                t0 = time.time()
                page_had_tables = False
                try:
                    if _page_has_table_indicators(page, page_text):
                        tables = page.extract_tables()
                        if tables:
                            stats["pages_with_tables"] += 1
                            page_had_tables = True
                            for table_num, table in enumerate(tables):
                                if not table or len(table) < 2:
                                    continue
                                table_id = f"table_{page_num+1}_{table_num+1}_{uuid.uuid4().hex[:8]}"
                                processed_table = _process_and_store_table(
                                    table, page_num + 1, table_num + 1, table_id
                                )
                                if processed_table:
                                    table_elements.append(Element(
                                        type="table",
                                        text=f"Table {page_num+1}.{table_num+1} (ID: {table_id}):\n{processed_table['display_text']}",
                                        page=page_num + 1,
                                    ))
                    else:
                        stats["pages_skipped_table_check"] += 1
                except Exception as e:
                    pass
                stats["table_time"] += time.time() - t0

                # Always include raw page text for pages with tables as a fallback
                if page_had_tables and page_text and page_text.strip():
                    text_elements.append(Element(
                        type="text",
                        text=f"Page {page_num + 1}, Raw Text (table fallback):\n\n{page_text.strip()}"
                    ))

                # Free page cache to reduce memory usage
                page.flush_cache()

                # --- Text chunking ---
                t0 = time.time()
                if not page_text or len(page_text.strip()) < 50:
                    if page_text:
                        _extract_text_fallback(page_text, page_num, text_elements)
                    stats["chunk_time"] += time.time() - t0
                    continue

                page_titles = _detect_titles_on_page(page_text, page_num)

                if page_titles:
                    _extract_text_with_titles(page_text, page_titles, page_num, text_elements)
                else:
                    _extract_text_with_paragraphs(page_text, page_num, text_elements)

                stats["processed_pages"].add(page_num + 1)
                stats["chunk_time"] += time.time() - t0


def extract_pdf_elements(pdf_path):
        """
        Extract text and tables from PDF using pdfplumber with title-based chunking.

        To keep memory bounded, the PDF is split into batches of
        ``PDF_BATCH_SIZE`` pages (default 50).  Each batch is written to a
        temporary file via *pypdf*, processed by pdfplumber independently,
        then freed with ``gc.collect()`` so that pdfminer's internal
        structures are released before the next batch starts.
        """
        text_elements = []
        table_elements = []

        start_time = time.time()
        stats = {
            "start_time": start_time,
            "text_time": 0.0,
            "table_time": 0.0,
            "chunk_time": 0.0,
            "pages_with_tables": 0,
            "pages_skipped_table_check": 0,
            "processed_pages": set(),
        }

        # --- Lightweight page-count check with pypdf (does NOT parse page content) ---
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        del reader  # free immediately

        if total_pages > PDF_MAX_PAGES:
            raise ValueError(
                f"PDF has {total_pages} pages, exceeding the limit of {PDF_MAX_PAGES}. "
                f"Set PDF_MAX_PAGES env var to increase the limit."
            )

        log.info(f"Processing PDF with {total_pages} pages (batch size: {PDF_BATCH_SIZE})")

        # --- Process in batches ---
        for batch_start in range(0, total_pages, PDF_BATCH_SIZE):
            batch_end = min(batch_start + PDF_BATCH_SIZE, total_pages)
            batch_num = (batch_start // PDF_BATCH_SIZE) + 1
            total_batches = (total_pages + PDF_BATCH_SIZE - 1) // PDF_BATCH_SIZE

            log.info(f"Processing batch {batch_num}/{total_batches} (pages {batch_start + 1}-{batch_end})")

            # Split the batch into a temporary PDF
            chunk_path = _split_pdf_to_temp(pdf_path, batch_start, batch_end)
            try:
                _extract_batch(
                    chunk_path, batch_start, total_pages,
                    text_elements, table_elements, stats,
                )
            finally:
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass

            # Force garbage collection to free pdfminer/pdfplumber internal structures
            gc.collect()

        total_time = time.time() - start_time
        log.info(
            f"PDF extraction complete: {total_pages} pages in {total_time:.1f}s "
            f"(text: {stats['text_time']:.1f}s, tables: {stats['table_time']:.1f}s, chunking: {stats['chunk_time']:.1f}s). "
            f"Found {len(table_elements)} tables on {stats['pages_with_tables']} pages, "
            f"{len(text_elements)} text chunks. "
            f"Skipped table detection on {stats['pages_skipped_table_check']}/{total_pages} pages."
        )

        return table_elements, text_elements
def _create_table_display_text_with_context(headers, data_rows, table_id, table_title, page_num, table_num):
        """
        Create enhanced display text that includes context about table position and content.
        Handles sub-table numbering and compound table context.
        """
        lines = []
        
        # Handle sub-table display
        if isinstance(table_num, str) and '.' in table_num:
            lines.append(f"Sub-Table {page_num}.{table_num} (ID: {table_id})")
            lines.append(f"Location: Page {page_num}, Sub-table {table_num}")
        else:
            lines.append(f"Table {page_num}.{table_num} (ID: {table_id})")
            lines.append(f"Location: Page {page_num}, Table #{table_num} on this page")
        
        if table_title:
            lines.append(f"Title: {table_title}")
        
        # Add content preview to help distinguish similar tables
        lines.append(f"Schema: {len(headers)} columns x {len(data_rows)} rows")
        lines.append(f"Columns: {', '.join(headers)}")
        
        # Add a few sample rows to show actual content
        lines.append("\nSample Data:")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---" for _ in headers]) + " |")
        
        # Show first few rows to distinguish content
        for i, row in enumerate(data_rows[:3]):  # Only show first 3 rows
            formatted_row = [str(cell) if cell else "" for cell in row]
            while len(formatted_row) < len(headers):
                formatted_row.append("")
            lines.append("| " + " | ".join(formatted_row[:len(headers)]) + " |")
        
        if len(data_rows) > 3:
            lines.append("| ... | (additional rows) | ... |")
        
        # Add content fingerprint to help distinguish similar tables
        content_sample = []
        for row in data_rows[:2]:  # Use first 2 rows for fingerprint
            for cell in row[:3]:  # Use first 3 cells
                if cell and str(cell).strip():
                    content_sample.append(str(cell).strip()[:10])  # First 10 chars
        
        if content_sample:
            lines.append(f"\nContent Fingerprint: {', '.join(content_sample[:6])}")
        
        # Add note if this is a sub-table
        if '_sub_' in table_id:
            lines.append(f"\nNote: This is a sub-table extracted from a larger compound table.")
        
        return "\n".join(lines)

def _process_and_store_table(table, page_num, table_num, table_id):
        """
        Process table data and store it in DuckDB for better querying.
        Handles nested tables (multiple logical tables within one extracted table).
        Each sub-table separated by titles will be stored as separate tables.
        """
        try:
            if not table or len(table) < 2:
                return None
            
            # Clean and process table data
            cleaned_table = []
            for row in table:
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                cleaned_table.append(cleaned_row)
            
            # Step 1: Detect if this is a compound table with multiple sub-tables
            sub_tables = _detect_and_split_compound_table(cleaned_table)
            
            existing_table = {}  # Replace with actual check if using a database
            
            if len(sub_tables) > 1:
                
                # Process each sub-table separately
                processed_tables = []
                for sub_idx, sub_table_info in enumerate(sub_tables):
                    sub_table_id = f"{table_id}_sub_{sub_idx + 1}"
                    
                    # Check if this sub-table already exists
                    # existing_table = conn.execute("""
                    #     SELECT table_id FROM table_metadata WHERE table_id = ?
                    # """, [sub_table_id]).fetchone()
                    
                    if sub_table_id in existing_table:
                        continue
                    
                    processed_sub_table = _process_single_table(
                        sub_table_info['data'], 
                        page_num, 
                        f"{table_num}.{sub_idx + 1}",  # Sub-table numbering
                        sub_table_id,
                        sub_table_info['title']
                    )
                    
                    if processed_sub_table:
                        existing_table[sub_table_id] = sub_table_id
                        processed_tables.append(processed_sub_table)
                
                return processed_tables[0] if processed_tables else None
            
            else:
                # Single table - process normally
                # existing_table = conn.execute("""
                #     SELECT table_id FROM table_metadata WHERE table_id = ?
                # """, [table_id]).fetchone()
                
                if table_id in existing_table:
                    return None
                processed_sub_table = _process_single_table(cleaned_table, page_num, table_num, table_id)
                if processed_sub_table:
                    existing_table[table_id] = table_id
                return processed_sub_table
            
        except Exception as e:
            log.error(f"Error processing table: {e}")
            import traceback
            traceback.print_exc()
            return None

def _detect_and_split_compound_table(cleaned_table):
        """
        Detect if a table contains multiple logical sub-tables separated by titles.
        IMPROVED: Better handling of tables with same headers but different data.
        Returns a list of sub-table information.
        """
        if not cleaned_table:
            return [{'data': cleaned_table, 'title': None}]
        
        # Look for potential separator rows (titles/headers between data sections)
        separator_rows = []
        potential_header_rows = []
        
        for row_idx, row in enumerate(cleaned_table):
            non_empty_cells = [cell for cell in row if cell.strip()]
            
            if not non_empty_cells:
                continue
            
            # FIRST: Detect potential header rows (repeated headers)
            is_potential_header = _looks_like_column_headers(non_empty_cells)
            if is_potential_header:
                potential_header_rows.append({
                    'row_idx': row_idx,
                    'headers': non_empty_cells,
                    'type': 'repeated_header'
                })
            
            # SECOND: Detect separator rows (titles between tables)
            is_separator = False
            separator_type = None
            
            # Skip if this looks like headers (we'll handle repeated headers separately)
            if is_potential_header:
                continue
            
            # 1. Single cell with content spanning multiple columns (merged title)
            if len(non_empty_cells) == 1 and len(row) > 1:
                cell_text = non_empty_cells[0].strip()
                # Make sure it's not just a data value
                if len(cell_text) > 3 and not _is_likely_numeric(cell_text):
                    is_separator = True
                    separator_type = "single_cell_title"
            
            # 2. Few cells filled, but very different from surrounding rows
            elif len(non_empty_cells) <= 3:
                # Check if this row looks different from adjacent rows
                prev_row = cleaned_table[row_idx - 1] if row_idx > 0 else []
                next_row = cleaned_table[row_idx + 1] if row_idx < len(cleaned_table) - 1 else []
                
                prev_filled = len([cell for cell in prev_row if cell.strip()])
                next_filled = len([cell for cell in next_row if cell.strip()])
                
                # If this row has significantly fewer filled cells than neighbors
                if (prev_filled > len(non_empty_cells) * 2 or next_filled > len(non_empty_cells) * 2):
                    # Additional check: make sure surrounding rows aren't also sparse
                    if prev_filled > 2 or next_filled > 2:
                        is_separator = True
                        separator_type = "sparse_title"
            
            # 3. Content pattern suggests it's a title
            row_text = ' '.join(non_empty_cells).lower()

            if any(indicator in row_text for indicator in _COMPOUND_TABLE_TITLE_INDICATORS):
                # Additional check: should not look like a data row
                if not any(cell.replace('.', '').replace(',', '').replace('$', '').isdigit() for cell in non_empty_cells):
                    is_separator = True
                    separator_type = "content_title"
            
            # 4. All caps or special formatting (common for titles)
            if len(non_empty_cells) <= 3:
                caps_count = sum(1 for cell in non_empty_cells if cell.isupper() and len(cell) > 2)
                if caps_count >= len(non_empty_cells) * 0.7:  # Most cells are caps
                    is_separator = True
                    separator_type = "caps_title"
            
            if is_separator:
                separator_rows.append({
                    'row_idx': row_idx,
                    'title': ' '.join(non_empty_cells),
                    'type': separator_type
                })
        
        # IMPROVED LOGIC: Handle repeated headers as table separators
        all_separators = []
        
        # Add title separators
        all_separators.extend(separator_rows)
        
        # Add repeated headers as separators (except the first occurrence)
        if len(potential_header_rows) > 1:
            first_header = potential_header_rows[0]['headers']
            
            for i, header_info in enumerate(potential_header_rows[1:], 1):
                current_headers = header_info['headers']
                
                # Check if headers are similar to the first occurrence
                if _headers_are_similar(first_header, current_headers):
                    all_separators.append({
                        'row_idx': header_info['row_idx'],
                        'title': f"Repeated table with headers: {', '.join(current_headers[:3])}...",
                        'type': 'repeated_header'
                    })
        
        # Sort all separators by row index
        all_separators.sort(key=lambda x: x['row_idx'])
        
        # If no separators found, treat as single table
        if not all_separators:
            return [{'data': cleaned_table, 'title': None}]
        
        # Split the table at separator rows
        sub_tables = []
        current_start = 0
        
        for separator in all_separators:
            separator_idx = separator['row_idx']
            
            # Get the section before this separator
            if separator_idx > current_start:
                section_data = cleaned_table[current_start:separator_idx]
                
                # Only include if it has substantial content
                if _has_substantial_table_content(section_data):
                    # For repeated headers, don't include the title since it's actually headers
                    if separator.get('type') == 'repeated_header':
                        sub_tables.append({
                            'data': section_data,
                            'title': None  # Previous section before repeated header
                        })
                    else:
                        sub_tables.append({
                            'data': section_data,
                            'title': None  # Previous sections don't get the separator as title
                        })
            
            # Handle the section after this separator
            current_start = separator_idx + 1
            
            # For repeated headers, include the header row in the next section
            if separator.get('type') == 'repeated_header':
                current_start = separator_idx  # Include the repeated header
            
            # Get the section after this separator
            if current_start < len(cleaned_table):
                # Find the next separator or end of table
                next_separator_idx = len(cleaned_table)
                for next_sep in all_separators:
                    if next_sep['row_idx'] > separator_idx:
                        next_separator_idx = next_sep['row_idx']
                        break
                
                section_data = cleaned_table[current_start:next_separator_idx]
                
                # Only include if it has substantial content
                if _has_substantial_table_content(section_data):
                    # For repeated headers, the title should indicate it's a continuation
                    if separator.get('type') == 'repeated_header':
                        sub_tables.append({
                            'data': section_data,
                            'title': f"Continuation table (same structure)"
                        })
                    else:
                        sub_tables.append({
                            'data': section_data,
                            'title': separator['title']  # This section gets the separator as its title
                        })
                
                current_start = next_separator_idx
        
        # Handle any remaining data after the last separator
        if current_start < len(cleaned_table):
            remaining_data = cleaned_table[current_start:]
            if _has_substantial_table_content(remaining_data):
                sub_tables.append({
                    'data': remaining_data,
                    'title': None
                })
        
        return sub_tables if sub_tables else [{'data': cleaned_table, 'title': None}]

def _headers_are_similar(headers1, headers2, similarity_threshold=0.7):
        """
        Check if two header rows are similar enough to be considered the same structure.
        """
        if len(headers1) != len(headers2):
            return False
        
        if len(headers1) == 0:
            return False
        
        matches = 0
        for h1, h2 in zip(headers1, headers2):
            h1_clean = str(h1).strip().lower()
            h2_clean = str(h2).strip().lower()
            
            if h1_clean == h2_clean:
                matches += 1
            elif h1_clean and h2_clean and (h1_clean in h2_clean or h2_clean in h1_clean):
                matches += 0.8  # Partial match
        
        similarity = matches / len(headers1)
        return similarity >= similarity_threshold

def _has_substantial_table_content(table_data):
        """
        Check if table data has substantial content to be considered a valid table.
        IMPROVED: Better detection of meaningful table content.
        """
        if not table_data or len(table_data) < 2:
            return False
        
        # Count rows with meaningful content
        substantial_rows = 0
        total_cells_with_content = 0
        
        for row in table_data:
            non_empty_cells = [cell for cell in row if cell and str(cell).strip()]
            
            if len(non_empty_cells) >= 2:  # At least 2 cells with content
                substantial_rows += 1
                total_cells_with_content += len(non_empty_cells)
        
        # Need at least 2 rows with content AND reasonable amount of data
        return substantial_rows >= 2 and total_cells_with_content >= 6

def _looks_like_column_headers(row_content):
        """
        Determine if a row looks like column headers.
        IMPROVED: Better detection to identify repeated headers.
        """
        if not row_content or len(row_content) < 2:
            return False
        
        # Count cells that look like column names
        header_like_count = 0
        
        for cell in row_content:
            cell_lower = str(cell).lower().strip()
            
            # Skip empty cells
            if not cell_lower:
                continue
            
            is_header_like = False

            # Check for header keywords (module-level frozenset)
            if any(keyword in cell_lower for keyword in _COLUMN_HEADER_KEYWORDS):
                is_header_like = True
            # Short descriptive text (typical of headers)
            elif (2 <= len(cell_lower) <= 25 and 
                any(c.isalpha() for c in cell_lower) and 
                not str(cell).replace(' ', '').replace('.', '').replace(',', '').isdigit() and
                cell_lower not in ['for', 'of', 'the', 'and', 'by', 'in', 'a', 'an']):  # Exclude common words
                is_header_like = True
            # Single letter or very short codes (often column headers)
            elif len(cell_lower) <= 3 and cell_lower.isalpha():
                is_header_like = True
            
            if is_header_like:
                header_like_count += 1
        
        # If most cells look like headers, this is probably a header row
        header_ratio = header_like_count / len(row_content) if row_content else 0
        
        return header_ratio >= 0.5  # 50% or more cells look like headers (more lenient for repeated headers)

def _process_single_table(cleaned_table, page_num, table_num, table_id, given_title=None):
        """
        Process a single logical table (may be a sub-table from a compound table).
        IMPROVED: Detect headers FIRST, then look for titles above headers.
        """
        try:
            if not cleaned_table or len(cleaned_table) < 2:
                return None
            
            # STEP 1: Find the header row first (most important)
            header_row_index = _find_header_row(cleaned_table)
            
            if header_row_index >= len(cleaned_table):
                return None
            
            # STEP 2: Extract headers
            headers = cleaned_table[header_row_index]
            
            # Validate headers before proceeding
            if not _validate_headers(headers):
                
                # Try the next row as headers if available
                if header_row_index + 1 < len(cleaned_table):
                    header_row_index += 1
                    headers = cleaned_table[header_row_index]
                    
                    if not _validate_headers(headers):
                        pass

            # STEP 3: Extract data rows (everything AFTER headers)
            data_rows = cleaned_table[header_row_index + 1:]
            
            # STEP 4: NOW look for titles ONLY in rows BEFORE the header row
            table_titles = []
            
            # Add given title first (from compound table splitting)
            if given_title:
                table_titles.append(given_title)
            
            # Look for titles ONLY in rows before the header row
            if header_row_index > 0:
                
                for row_idx in range(header_row_index):
                    row_content = [cell for cell in cleaned_table[row_idx] if cell.strip()]
                    
                    if row_content:
                        row_text = ' '.join(row_content)
                        
                        # ONLY include if it genuinely looks like a title (not headers)
                        if _looks_like_title(row_text, row_content):
                            table_titles.append(row_text)

            
            # STEP 5: Create final title
            table_title = '; '.join(table_titles) if table_titles else None
            
            # STEP 6: Filter out empty data rows
            filtered_data_rows = []
            for row in data_rows:
                has_content = any(cell.strip() for cell in row if cell)
                if has_content:
                    filtered_data_rows.append(row)
            
            if not filtered_data_rows:
                return None
            
            
            # STEP 7: Clean headers and ensure uniqueness
            clean_headers = []
            seen_headers = set()
            
            for i, header in enumerate(headers):
                if not header or header.isspace():
                    base_name = f"column_{i+1}"
                else:
                    clean_header = re.sub(r'[^\w\s]', '', str(header))
                    base_name = re.sub(r'\s+', '_', clean_header).lower() or f"column_{i+1}"
                
                final_name = base_name
                counter = 1
                while final_name in seen_headers:
                    final_name = f"{base_name}_{counter}"
                    counter += 1
                
                clean_headers.append(final_name)
                seen_headers.add(final_name)
            
            # STEP 8: Store in database
            try:
                # Calculate table position score (for ordering on the same page)
                if isinstance(table_num, str) and '.' in table_num:
                    main_num, sub_num = table_num.split('.', 1)
                    position_score = (page_num * 1000) + (int(main_num) * 10) + int(sub_num)
                else:
                    position_score = (page_num * 1000) + int(table_num)
                
                
                # Store table data
                for row_idx, row in enumerate(filtered_data_rows):
                    for col_idx, cell_value in enumerate(row):
                        if col_idx < len(clean_headers):
                            column_name = clean_headers[col_idx]
                            
                            if cell_value and str(cell_value).strip():
                                cell_type = _determine_cell_type(str(cell_value))
                                
                
                
            except Exception as db_error:
                log.error(f"Database error storing table {table_id}: {db_error}")
                # try:
                #     conn.execute("DELETE FROM table_data WHERE table_id = ?", [table_id])
                #     # conn.execute("DELETE FROM table_metadata WHERE table_id = ?", [table_id])
                # except:
                #     pass
                return None
            
            # STEP 9: Create display text
            display_text = _create_table_display_text_with_context(
                clean_headers, filtered_data_rows, table_id, table_title, page_num, table_num
            )
            
            return {
                "table_id": table_id,
                "headers": clean_headers,
                "data_rows": filtered_data_rows,
                "display_text": display_text,
                "title": table_title,
                "page": page_num,
                "table_num": table_num
            }
            
        except Exception as e:
            log.error(f"Error processing single table: {e}")
            import traceback
            traceback.print_exc()
            return None

def _looks_like_title(row_text, row_content):
        """
        Determine if a row looks like a title rather than headers or data.
        IMPROVED: More strict criteria to avoid false positives.
        """
        if not row_text or not row_content:
            return False
        
        # RULE 1: Single cell spanning multiple columns (classic title pattern)
        if len(row_content) == 1:
            cell_text = row_content[0].strip()
            
            # Must be substantial text (not just a number or single word)
            if len(cell_text) < 3:
                return False
            
            # Title indicators
            title_patterns = [
                'table', 'summary', 'overview', 'results', 'analysis', 'data',
                'comparison', 'statistics', 'breakdown', 'distribution', 'report',
                'section', 'part', 'chapter', 'appendix', 'figure', 'chart',
                'year', 'quarter', 'month', 'period', 'fiscal', 'annual'
            ]
            
            cell_lower = cell_text.lower()
            
            # Check for title patterns
            if any(pattern in cell_lower for pattern in title_patterns):
                return True
            
            # All caps (often used for titles) - but not single words
            if cell_text.isupper() and len(cell_text) > 8 and ' ' in cell_text:
                return True
            
            # Contains descriptive connecting words
            if any(word in cell_lower for word in ['for', 'of', 'the', 'and', 'by', 'in', 'during', 'from', 'to']):
                return True
            
            # Long descriptive text (likely a title)
            if len(cell_text) > 25 and not cell_text.replace(' ', '').isdigit():
                return True
        
        # RULE 2: Two or three cells that form a descriptive phrase
        elif 2 <= len(row_content) <= 3:
            combined_text = ' '.join(row_content).strip()
            
            # Skip if it looks like headers
            if _looks_like_column_headers(row_content):
                return False
            
            # Check if it forms a coherent title phrase
            if (len(combined_text) > 15 and 
                any(word in combined_text.lower() for word in ['for', 'of', 'the', 'and', 'by', 'during', 'from']) and
                not any(word in combined_text.lower() for word in ['name', 'value', 'amount', 'count', 'total'])):
                return True
        
        # RULE 3: Reject if it has too many cells (likely headers)
        if len(row_content) > 3:
            return False
        
        # RULE 4: Reject if it looks like data
        # Check if most cells are numbers
        numeric_count = 0
        for cell in row_content:
            if _is_likely_numeric(str(cell)):
                numeric_count += 1
        
        if numeric_count / len(row_content) > 0.5:  # More than half are numbers
            return False
        
        return False

def _looks_like_column_headers(row_content):
        """
        Determine if a row looks like column headers.
        IMPROVED: Better detection to avoid confusion with titles.
        """
        if not row_content or len(row_content) < 2:
            return False
        
        # Count cells that look like column names
        header_like_count = 0
        
        for cell in row_content:
            cell_lower = str(cell).lower().strip()
            
            # Skip empty cells
            if not cell_lower:
                continue
            
            # Common column header patterns
            header_keywords = [
                'name', 'type', 'value', 'amount', 'date', 'time', 'category', 'id', 'number',
                'count', 'total', 'sum', 'average', 'mean', 'min', 'max', 'year', 'month', 'day',
                'description', 'title', 'status', 'code', 'rate', 'price', 'cost', 'quantity',
                'size', 'weight', 'length', 'width', 'height', 'area', 'volume', 'percent',
                'percentage', 'ratio', 'index', 'score', 'rank', 'level', 'grade', 'class',
                'item', 'product', 'service', 'customer', 'order', 'invoice', 'payment'
            ]
            
            # Check for header keywords
            if any(keyword in cell_lower for keyword in header_keywords):
                header_like_count += 1
            # Short descriptive text (typical of headers)
            elif (3 <= len(cell_lower) <= 25 and 
                any(c.isalpha() for c in cell_lower) and 
                not str(cell).replace(' ', '').isdigit() and
                cell_lower not in ['for', 'of', 'the', 'and', 'by', 'in']):  # Exclude title words
                header_like_count += 1
        
        # If most cells look like headers, this is probably a header row
        header_ratio = header_like_count / len(row_content) if row_content else 0
        
        
        return header_ratio >= 0.6  # 60% or more cells look like headers

def _validate_headers(headers):
        """
        Validate that the detected headers look reasonable.
        IMPROVED: Better validation to catch obvious non-header rows.
        """
        if not headers:
            return False
        
        # Count non-empty headers
        non_empty_headers = [h for h in headers if h and str(h).strip()]
        
        if len(non_empty_headers) < 2:  # Need at least 2 columns
            return False
        
        # Check for obviously bad headers
        bad_headers = 0
        total_non_empty = len(non_empty_headers)
        
        for header in non_empty_headers:
            header_str = str(header).strip()
            
            # Skip empty headers
            if not header_str:
                continue
            
            # Bad header patterns
            is_bad = (
                header_str.isdigit() or  # Pure numbers (except years)
                (len(header_str) == 1 and not header_str.isalpha()) or  # Single non-letter chars
                header_str in ['', ' ', '-', '|', '_', '.', ','] or  # Meaningless chars
                len(header_str) > 100 or  # Too long (probably data, not header)
                # Obvious data patterns
                ('$' in header_str and any(c.isdigit() for c in header_str)) or
                (header_str.count('.') > 1) or  # Multiple decimals
                (header_str.count(',') > 2)  # Too many commas
            )
            
            # Special case: 4-digit numbers might be years (acceptable headers)
            if header_str.isdigit() and len(header_str) == 4 and 1900 <= int(header_str) <= 2100:
                is_bad = False
            
            if is_bad:
                bad_headers += 1
        
        # If more than half the headers look bad, this is probably not a header row
        bad_ratio = bad_headers / total_non_empty if total_non_empty > 0 else 1
        
        
        return bad_ratio <= 0.4  # Allow up to 40% bad headers (some tables have weird formatting)

def _find_header_row(cleaned_table):
        """
        Find the most likely header row in the table using multiple heuristics.
        IMPROVED: Detect repeated headers and prefer the first occurrence.
        Returns the index of the row that should be treated as headers.
        """
        if not cleaned_table:
            return 0
        
        # STEP 1: First identify all potential header rows
        potential_headers = []
        
        for row_idx, row in enumerate(cleaned_table):
            non_empty_cells = [cell for cell in row if cell and str(cell).strip()]
            
            if len(non_empty_cells) >= 2 and _looks_like_column_headers(non_empty_cells):
                score = _calculate_header_score(row, row_idx, cleaned_table)
                potential_headers.append({
                    'row_idx': row_idx,
                    'headers': non_empty_cells,
                    'score': score,
                    'full_row': row
                })
        
        if not potential_headers:
            # Fallback to original scoring if no clear headers found
            return _find_header_row_fallback(cleaned_table)
        
        # STEP 2: Check for repeated headers (same column names in different rows)
        header_groups = {}
        
        for header_info in potential_headers:
            # Create a signature based on the first few column names
            core_headers = header_info['headers'][:3]  # Use first 3 columns as signature
            core_signature = tuple(str(h).lower().strip() for h in core_headers)
            
            if core_signature not in header_groups:
                header_groups[core_signature] = []
            header_groups[core_signature].append(header_info)
        
        # STEP 3: Handle repeated headers - prefer the FIRST occurrence
        selected_header_row = None
        
        for signature, group in header_groups.items():
            if len(group) > 1:
                # Multiple rows with same header signature = repeated headers
                
                # Sort by row index and pick the FIRST one
                group.sort(key=lambda x: x['row_idx'])
                first_occurrence = group[0]
                
                
                # Validate the first occurrence
                if _validate_headers(first_occurrence['full_row']):
                    selected_header_row = first_occurrence['row_idx']
                    break
                else:
                    for header_info in group[1:]:
                        if _validate_headers(header_info['full_row']):
                            selected_header_row = header_info['row_idx']
                            break
                    if selected_header_row:
                        break
        
        # STEP 4: If no repeated headers, pick the best single header row
        if selected_header_row is None:
            # Sort by score and validate
            potential_headers.sort(key=lambda x: x['score'], reverse=True)
            
            for header_info in potential_headers:
                if _validate_headers(header_info['full_row']):
                    selected_header_row = header_info['row_idx']
                    break
        
        # STEP 5: Final validation and fallback
        if selected_header_row is not None:
            return selected_header_row
        else:
            return _find_header_row_fallback(cleaned_table)

def _calculate_header_score(row, row_idx, cleaned_table):
        """Calculate score for a potential header row."""
        score = 0
        non_empty_cells = [cell for cell in row if cell and str(cell).strip()]
        
        if not non_empty_cells:
            return -100
        
        # 1. Number of non-empty cells (headers usually fill most columns)
        filled_ratio = len(non_empty_cells) / len(row) if len(row) > 0 else 0
        score += filled_ratio * 30  # Increased weight
        
        # 2. Text characteristics that suggest headers
        header_keywords = [
            'name', 'type', 'value', 'amount', 'date', 'time', 'category', 'id', 'number',
            'count', 'total', 'sum', 'average', 'mean', 'min', 'max', 'year', 'month', 'day',
            'description', 'title', 'status', 'code', 'rate', 'price', 'cost', 'quantity',
            'size', 'weight', 'length', 'width', 'height', 'area', 'volume', 'percent',
            'percentage', 'ratio', 'index', 'score', 'rank', 'level', 'grade', 'class',
            'placement', 'material', 'supplier', 'article', 'stitch', 'gauge', 'ends'  # Domain-specific
        ]
        
        header_score = 0
        numeric_cells = 0
        
        for cell in non_empty_cells:
            cell_str = str(cell).strip()
            cell_lower = cell_str.lower()
            
            # Strong header indicators
            if any(keyword in cell_lower for keyword in header_keywords):
                header_score += 15
            
            # Good length for headers
            if 3 <= len(cell_str) <= 25:
                header_score += 8
            
            # Contains letters
            if any(c.isalpha() for c in cell_str):
                header_score += 5
            
            # Heavy penalty for purely numeric cells
            if cell_str.replace('.', '').replace(',', '').replace('-', '').isdigit():
                numeric_cells += 1
                header_score -= 20
            
            # Penalty for very long text
            if len(cell_str) > 50:
                header_score -= 10
            
            # Penalty for obvious data patterns
            if '$' in cell_str or '%' in cell_str:
                header_score -= 10
        
        # Heavy penalty if most cells are numeric
        if len(non_empty_cells) > 0 and numeric_cells / len(non_empty_cells) > 0.5:
            header_score -= 100
        
        score += header_score
        
        # 3. Position-based scoring (prefer earlier rows, but not too heavily)
        position_score = max(0, 15 - row_idx * 2)
        score += position_score
        
        # 4. Check difference with next row
        if row_idx + 1 < len(cleaned_table):
            next_row = cleaned_table[row_idx + 1]
            next_non_empty = [cell for cell in next_row if cell and str(cell).strip()]
            
            if next_non_empty:
                current_numeric_ratio = sum(1 for cell in non_empty_cells if _is_likely_numeric(str(cell))) / len(non_empty_cells)
                next_numeric_ratio = sum(1 for cell in next_non_empty if _is_likely_numeric(str(cell))) / len(next_non_empty)
                
                if next_numeric_ratio > current_numeric_ratio + 0.3:
                    score += 25
        
        # 5. Consistency check
        if non_empty_cells:
            avg_length = sum(len(str(cell)) for cell in non_empty_cells) / len(non_empty_cells)
            if 5 <= avg_length <= 25:
                score += 10
        
        # 6. Check for title-like patterns (penalize)
        row_text = ' '.join(str(cell) for cell in non_empty_cells).lower()
        title_indicators = ['bom:', 'section:', 'colorway', 'part', 'style', 'design']
        
        if any(indicator in row_text for indicator in title_indicators) and len(non_empty_cells) <= 3:
            score -= 30
        
        return score

def _find_header_row_fallback(cleaned_table):
        """Fallback method using original logic."""
        # Original scoring logic as backup
        row_scores = []
        
        for row_idx, row in enumerate(cleaned_table):
            score = 0
            non_empty_cells = [cell for cell in row if cell and str(cell).strip()]
            
            if not non_empty_cells:
                row_scores.append(-100)
                continue
            
            if len(non_empty_cells) < 2:
                row_scores.append(-50)
                continue
            
            # Basic scoring
            filled_ratio = len(non_empty_cells) / len(row) if len(row) > 0 else 0
            score += filled_ratio * 20
            
            # Header characteristics
            for cell in non_empty_cells:
                cell_lower = str(cell).lower()
                if any(keyword in cell_lower for keyword in ['name', 'type', 'value', 'material', 'supplier']):
                    score += 10
                if 3 <= len(str(cell)) <= 30 and not str(cell).isdigit():
                    score += 5
                if any(c.isalpha() for c in str(cell)):
                    score += 3
                if str(cell).replace('.', '').replace(',', '').isdigit():
                    score -= 5
            
            # Position preference
            position_score = max(0, 10 - row_idx * 2)
            score += position_score
            
            row_scores.append(score)
        
        if not row_scores:
            return 0
        
        best_row_idx = row_scores.index(max(row_scores))
        return best_row_idx

def _is_likely_numeric(cell_value):
        """Helper method to check if a cell value looks numeric."""
        if not cell_value:
            return False
        
        # Remove common formatting characters
        cleaned = cell_value.replace(',', '').replace('$', '').replace('%', '').replace(' ', '')
        
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

def _determine_cell_type(cell_value):
        """Determine the type of a cell value."""
        if not cell_value or cell_value.isspace():
            return "empty"
        
        # Try to parse as number
        try:
            float(cell_value.replace(',', '').replace('$', '').replace('%', ''))
            return "number"
        except ValueError:
            pass
        
        # Check for date patterns
        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', cell_value):
            return "date"
        
        return "text"
        
def summarize_elements(elements):
        """Generate ONE overall summary for ALL elements in the document for maximum performance."""
        if not elements:
            return [], []
        
        # Separate tables and text for different handling
        table_elements = [e for e in elements if e.type == "table"]
        text_elements = [e for e in elements if e.type == "text"]
        
        
        # Create one comprehensive document overview
        document_content = _create_comprehensive_document_overview(text_elements, table_elements)
        
        # Generate ONE summary for the entire document
        overall_summary = _generate_single_document_summary(document_content, len(text_elements), len(table_elements))
        
        # Return the original texts with the SAME summary for all elements
        # This allows the existing retrieval system to work while being much more efficient
        results_texts = []
        results_summaries = []
        
        # Add text elements with shared summary
        for element in text_elements:
            results_texts.append(element.text)
            results_summaries.append(overall_summary)  # Same summary for all
        
        # Add table elements with enhanced shared summary
        for element in table_elements:
            # Extract table ID for context
            table_id_match = re.search(r'ID: ([^)]+)', element.text)
            table_id = table_id_match.group(1) if table_id_match else "unknown"
            
            # Add table-specific context to the shared summary
            table_enhanced_summary = f"{overall_summary}\n\n[TABLE CONTEXT: This specific table has ID {table_id} and contains structured data that can be queried directly from the database.]"
            
            results_texts.append(element.text)
            results_summaries.append(table_enhanced_summary)
        
        return results_texts, results_summaries

def _create_comprehensive_document_overview(text_elements, table_elements):
        """Create a comprehensive overview of the entire document content."""
        
        # Analyze text elements for structure and content
        text_overview = []
        page_sections = {}
        
        for element in text_elements:
            # Extract page and section information
            page_match = re.search(r'Page (\d+)', element.text)
            section_match = re.search(r'Section: ([^,\n]+)', element.text)
            
            page_num = page_match.group(1) if page_match else "Unknown"
            section_name = section_match.group(1) if section_match else "Content"
            
            if page_num not in page_sections:
                page_sections[page_num] = []
            
            # Get first few sentences for overview
            content_lines = element.text.split('\n')
            content_preview = []
            for line in content_lines[2:]:  # Skip page/section headers
                if line.strip() and len(' '.join(content_preview)) < 200:
                    content_preview.append(line.strip())
                if len(' '.join(content_preview)) >= 200:
                    break
            
            preview_text = ' '.join(content_preview)[:200] + "..." if content_preview else "No content preview"
            
            page_sections[page_num].append({
                'section': section_name,
                'preview': preview_text
            })
        
        # Create text structure summary
        for page_num in sorted(page_sections.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            sections = page_sections[page_num]
            text_overview.append(f"Page {page_num}: {len(sections)} sections")
            for section in sections[:3]:  # Limit to first 3 sections per page
                text_overview.append(f"  - {section['section']}: {section['preview']}")
            if len(sections) > 3:
                text_overview.append(f"  - ... and {len(sections) - 3} more sections")
        
        # Analyze table elements
        table_overview = []
        table_summary_info = {}
        
        for element in table_elements:
            # Extract table metadata
            table_id_match = re.search(r'ID: ([^)]+)', element.text)
            page_match = re.search(r'Table (\d+)\.(\d+)', element.text)
            
            table_id = table_id_match.group(1) if table_id_match else "unknown"
            page_num = page_match.group(1) if page_match else "Unknown"
            table_num = page_match.group(2) if page_match else "Unknown"
            
            # Get table structure from DuckDB if available
            table_info = f"Table {page_num}.{table_num} (ID: {table_id})"
            
            try:
                # metadata = conn.execute("""
                #     SELECT column_names, row_count, table_title
                #     FROM table_metadata 
                #     WHERE table_id = ?
                # """, [table_id]).fetchone()
                metadata = None  # Placeholder for actual metadata retrieval    
                if metadata:
                    column_names, row_count, table_title = metadata
                    table_info += f" - {table_title if table_title else 'No title'}"
                    table_info += f" - {len(column_names)} columns: {', '.join(column_names[:3])}"
                    if len(column_names) > 3:
                        table_info += f" and {len(column_names) - 3} more"
                    table_info += f" - {row_count} rows of data"
                else:
                    table_info += " - Structure details not available"
                    
            except Exception as e:
                table_info += " - Could not retrieve structure details"
            
            table_overview.append(table_info)
        
        # Combine everything into comprehensive overview
        document_overview = f"""
    DOCUMENT STRUCTURE ANALYSIS:

    TEXT CONTENT OVERVIEW ({len(text_elements)} sections):
    {chr(10).join(text_overview)}

    TABLE CONTENT OVERVIEW ({len(table_elements)} tables):
    {chr(10).join(table_overview)}

    FULL DOCUMENT SCOPE:
    Total pages analyzed: {len(set(page_sections.keys()))}
    Total text sections: {len(text_elements)}
    Total tables: {len(table_elements)}
    Document appears to contain: {"text with data tables" if table_elements else "primarily text content"}
    """
        
        return document_overview

def _generate_single_document_summary(document_overview, num_text_elements, num_table_elements):
        """Generate one comprehensive summary for the entire document."""
        
        summary_prompt = f"""You are an expert document analyst. Based on the comprehensive document overview below, create ONE master summary that covers the entire document.

    {document_overview}

    Please provide a comprehensive summary that includes:

    1. **Document Purpose & Scope**: What this document is about and what it covers
    2. **Main Content Areas**: The key topics, sections, or themes discussed
    3. **Data & Tables**: Overview of any tabular data, statistics, or structured information
    4. **Key Findings**: Important conclusions, results, or insights
    5. **Document Structure**: How the content is organized (pages, sections, etc.)
    6. **Searchable Topics**: What kinds of questions this document could answer

    Format your response as a well-structured summary that would help someone understand:
    - What information is available in this document
    - What questions they could ask about it
    - Where to find specific types of information (text vs tables)
    - The overall value and content of the document

    This summary will be used to help users understand what information is available and to improve search results across all {num_text_elements + num_table_elements} document elements.

    COMPREHENSIVE DOCUMENT SUMMARY:"""

        try:
            # Build the summary LLM per LLM_DEFAULT_PROVIDER (override with
            # PDF_SUMMARY_PROVIDER / PDF_SUMMARY_MODEL). Note: the previous code
            # referenced an undefined `api_key` here (NameError); the openai path
            # now falls back to OPENAI_API_KEY from config.
            from open_webui.llm.utils.task_llm import build_task_llm

            model = build_task_llm(
                "PDF_SUMMARY", default_openai_model="gpt-4o", temperature=0
            )
            response = model.invoke(summary_prompt)
            
            summary = response.content
            
            # Add metadata about the document structure
            metadata_addition = f"""

    [DOCUMENT METADATA]
    - Text sections: {num_text_elements}
    - Tables: {num_table_elements}  
    - Total elements: {num_text_elements + num_table_elements}
    - Processing date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}

    This summary applies to all document elements and provides context for search and retrieval operations."""

            return summary + metadata_addition
            
        except Exception as e:
            log.error(f"Error generating document summary: {e}")
            
            # Fallback summary
            return f"""Document Summary (Auto-generated):

    This document contains {num_text_elements} text sections and {num_table_elements} tables across multiple pages. 

    The content includes both textual information and structured data in table format. Users can search for specific topics, ask questions about the content, or query data from the tables.

    [Error generating detailed summary: {str(e)}]

    Document processing completed with {num_text_elements + num_table_elements} total elements."""
    
def add_to_retriever(texts, summaries, element_type):
        """Add texts and their summaries to the retriever."""
        # Safety check for empty inputs
        if not texts or not summaries or len(texts) == 0 or len(summaries) == 0:
            return 0
            
        ids = [str(uuid.uuid4()) for _ in texts]
        summary_docs = [
            Document(page_content=s, metadata={id_key: ids[i], "type": element_type})
            for i, s in enumerate(summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(ids, texts)))
        
        return len(ids)
    
def embedPDF(pdf_file: BinaryIO, pdf_name: str = None) -> Dict:
        """
        Embed a PDF file into the vector database using pdfplumber instead of unstructured.
        
        Args:
            pdf_file: The PDF file as a binary stream
            pdf_name: Optional name for the PDF
            
        Returns:
            Dict with embedding statistics
        """
        start_time = time.time()
        log.info(f"embedPDF started for: {pdf_name}")

        # Create a temporary file to process
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_path = temp_pdf.name

        try:
            # Validate file size before processing
            file_size = os.path.getsize(temp_path)
            if file_size > PDF_MAX_FILE_SIZE:
                raise ValueError(
                    f"PDF '{pdf_name}' is {file_size / 1024 / 1024:.1f}MB, "
                    f"exceeding the {PDF_MAX_FILE_SIZE_MB}MB limit. "
                    f"Set PDF_MAX_FILE_SIZE_MB env var to increase the limit."
                )

            # Extract elements from PDF using pdfplumber
            table_elements, text_elements = extract_pdf_elements(temp_path)

            elapsed = time.time() - start_time
            log.info(
                f"embedPDF completed for {pdf_name} in {elapsed:.1f}s: "
                f"{len(table_elements) if table_elements else 0} tables, "
                f"{len(text_elements) if text_elements else 0} text chunks"
            )

            # Generate summaries for all tables in batches
            if table_elements:
                table_elements = generate_table_summaries_batch(table_elements)

            return {
                "status": "success",
                "pdf_name": pdf_name,
                "tables_embedded": len(table_elements) if table_elements else 0,
                "text_chunks_embedded": len(text_elements) if text_elements else 0,
                "table_elements": table_elements,
                "text_elements": text_elements
            }

        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

def _table_to_markdown(table) -> str:
        """Convert a fitz Table object to a markdown string."""
        data = table.extract()
        if not data:
            return ""
        lines = []
        for i, row in enumerate(data):
            cells = [str(c) if c is not None else "" for c in row]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(lines)

def extract_pdf_with_langchain_loader(pdf_file: BinaryIO, pdf_name: str = None) -> Dict:
        """
        Extract PDF content using PyMuPDF (fitz) with table/text separation.
        Tables are extracted via page.find_tables() and converted to markdown.
        Text is extracted separately with tables excluded from the text blocks.
        Returns the same dict shape as embedPDF for API compatibility.
        """
        import fitz

        start_time = time.time()
        log.info(f"extract_pdf_with_langchain_loader started for: {pdf_name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_path = temp_pdf.name

        try:
            file_size = os.path.getsize(temp_path)
            if file_size > PDF_MAX_FILE_SIZE:
                raise ValueError(
                    f"PDF '{pdf_name}' is {file_size / 1024 / 1024:.1f}MB, "
                    f"exceeding the {PDF_MAX_FILE_SIZE_MB}MB limit. "
                    f"Set PDF_MAX_FILE_SIZE_MB env var to increase the limit."
                )

            doc = fitz.open(temp_path)
            total_pages = len(doc)

            if total_pages > PDF_MAX_PAGES:
                doc.close()
                raise ValueError(
                    f"PDF has {total_pages} pages, exceeding the limit of {PDF_MAX_PAGES}. "
                    f"Set PDF_MAX_PAGES env var to increase the limit."
                )

            table_elements = []
            text_elements = []

            for page in doc:
                page_num = page.number + 1

                # Extract tables and get their bounding boxes
                tab_finder = page.find_tables()
                table_rects = []
                for table in tab_finder.tables:
                    md = _table_to_markdown(table)
                    if md:
                        table_elements.append(Element(type="table", text=md, page=page_num))
                        table_rects.append(fitz.Rect(table.bbox))

                # Extract text, excluding table regions
                text_parts = []
                if table_rects:
                    # Get text blocks and filter out those inside table bounding boxes
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        block_rect = fitz.Rect(block[:4])
                        in_table = any(block_rect.intersects(tr) for tr in table_rects)
                        if not in_table:
                            block_text = block[4].strip()
                            if block_text:
                                text_parts.append(block_text)
                else:
                    page_text = page.get_text().strip()
                    if page_text:
                        text_parts.append(page_text)

                if text_parts:
                    text_elements.append(Element(
                        type="text",
                        text="\n\n".join(text_parts),
                        page=page_num,
                    ))

            doc.close()

            elapsed = time.time() - start_time
            log.info(
                f"extract_pdf_with_langchain_loader completed for {pdf_name} in {elapsed:.1f}s: "
                f"{len(table_elements)} tables, {len(text_elements)} text chunks"
            )

            # Generate summaries for all tables in batches
            if table_elements:
                table_elements = generate_table_summaries_batch(table_elements)

            return {
                "status": "success",
                "pdf_name": pdf_name,
                "tables_embedded": len(table_elements),
                "text_chunks_embedded": len(text_elements),
                "table_elements": table_elements,
                "text_elements": text_elements,
            }

        finally:
            os.unlink(temp_path)