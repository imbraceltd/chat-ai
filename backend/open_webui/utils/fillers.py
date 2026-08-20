async def pdf_filler(**kwargs) -> str:
    """Fills a PDF and returns a URL to the new file."""
    print(f"Filling PDF at {kwargs.get('fileUrlToFill')} with data.")
    return "https://example.com/filled_document.pdf"

async def excel_filler(**kwargs) -> str:
    """Fills an Excel file and returns a URL."""
    print(f"Filling Excel file at {kwargs.get('fileUrlToFill')} with data.")
    return "https://example.com/filled_document.xlsx"