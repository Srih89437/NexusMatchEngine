import logging
from pathlib import Path
from typing import Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoclingLayoutParser:
    """Layout-aware document parser targeting multi-column PDFs/DOCX using Docling with OCR."""

    def __init__(self):
        logger.info("Initializing Docling layout-aware document converter with OCR...")
        try:
            # Enable native OCR via PdfPipelineOptions
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("Docling DocumentConverter with OCR initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Docling: {e}")
            raise RuntimeError(f"Docling initialization failed: {e}") from e

    def parse_document(self, file_path: Path) -> Dict[str, Any]:
        """Extract layout elements, tables, and structures from unstructured files.

        Args:
            file_path: Path to target PDF, DOCX, or Markdown document.

        Returns:
            Dict containing raw text paragraphs (represented as markdown), tables, and headers.
        """
        logger.info(f"Parsing document at: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = {
            "file_name": file_path.name,
            "file_type": file_path.suffix,
            "size_bytes": file_path.stat().st_size,
        }

        try:
            # Convert using Docling layout analyzer (includes OCR)
            result = self.converter.convert(str(file_path))
            doc = result.document

            # Export to markdown to preserve layout semantics
            markdown_content = doc.export_to_markdown()

            # Extract headings and tables from document layout items
            tables = []
            headers = []
            for item, _ in doc.iterate_items():
                label = getattr(item, "label", None)
                text = getattr(item, "text", None)
                if label == "table":
                    tables.append(str(text or item))
                elif label == "section_header" and text:
                    headers.append(text)
        except Exception as docling_err:
            logger.warning(
                f"Docling failed to parse {file_path}, attempting fallback parsing: {docling_err}"
            )
            tables = []
            headers = []
            markdown_content = ""

            # Simple text/fallback parser
            suffix = file_path.suffix.lower()
            if suffix in [".txt", ".md", ".json"]:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        markdown_content = f.read()
                except Exception as e:
                    logger.error(f"Text fallback parser failed: {e}")
                    raise docling_err
            elif suffix == ".pdf":
                try:
                    import pypdfium2 as pdfium

                    pdf = pdfium.PdfDocument(str(file_path))
                    text_parts = []
                    for page in pdf:
                        textpage = page.get_textpage()
                        text_parts.append(textpage.get_text_range())
                    markdown_content = "\n".join(text_parts)
                except Exception as e:
                    logger.error(f"PDF fallback parser failed: {e}")
                    raise docling_err
            else:
                # Raise the original error if no fallback is possible
                raise docling_err

        return {
            "metadata": metadata,
            "raw_text": markdown_content,
            "tables": tables,
            "headers": headers,
        }
