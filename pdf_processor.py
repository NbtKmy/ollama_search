"""PDF Processing Module for Document Summarization

This module provides functionality to:
- Extract text from PDFs
- Analyze document structure (headings, sections)
- Chunk documents intelligently
- Support hybrid summarization strategies
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import pdfplumber


@dataclass
class PDFSection:
    """Represents a section in a PDF document."""
    title: str
    content: str
    level: int  # Heading level (1=chapter, 2=section, etc.)
    page_start: int
    page_end: int


@dataclass
class PDFChunk:
    """Represents a chunk of PDF content."""
    content: str
    chunk_index: int
    section_title: Optional[str] = None
    page_start: int = 0
    page_end: int = 0


@dataclass
class PDFDocument:
    """Represents a processed PDF document."""
    text: str
    num_pages: int
    has_structure: bool
    sections: List[PDFSection]
    metadata: dict


class PDFProcessor:
    """Process PDF documents for summarization."""

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Target number of words per chunk
        """
        self.chunk_size = chunk_size

    def extract_text(self, pdf_path: str) -> PDFDocument:
        """
        Extract text and structure from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFDocument with extracted text and metadata
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        full_text = []
        sections = []
        metadata = {}

        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata or {}
            num_pages = len(pdf.pages)

            # Extract text from all pages
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                full_text.append(text)

                # Try to detect headings based on font size
                # This is a simple heuristic - can be improved
                sections.extend(self._detect_sections_in_page(
                    page, page_num, text
                ))

        full_document_text = "\n\n".join(full_text)
        has_structure = len(sections) > 0

        return PDFDocument(
            text=full_document_text,
            num_pages=num_pages,
            has_structure=has_structure,
            sections=sections,
            metadata=metadata
        )

    def _detect_sections_in_page(
        self, page, page_num: int, text: str
    ) -> List[PDFSection]:
        """
        Detect sections/headings in a page.

        Uses font size and text patterns to identify headings.
        """
        sections = []

        try:
            # Get all text with formatting info
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=True
            )

            if not words:
                return sections

            # Calculate average font size
            font_sizes = [float(w.get('height', 0)) for w in words if w.get('height')]
            if not font_sizes:
                return sections

            avg_font_size = sum(font_sizes) / len(font_sizes)

            # Group words into lines
            lines = self._group_words_into_lines(words)

            # Detect headings (larger than average font + certain patterns)
            for line in lines:
                if not line['words']:
                    continue

                line_text = line['text'].strip()
                max_font_size = max(w.get('height', 0) for w in line['words'])

                # Heading heuristics
                is_larger = max_font_size > avg_font_size * 1.2
                is_short = len(line_text.split()) < 15
                looks_like_heading = (
                    re.match(r'^(\d+\.?\s+|[A-Z]\.?\s+|Chapter\s+\d+)', line_text) or
                    line_text.isupper() or
                    (is_short and line_text and line_text[0].isupper())
                )

                if is_larger and looks_like_heading:
                    # Determine heading level based on font size
                    if max_font_size > avg_font_size * 1.5:
                        level = 1  # Chapter
                    else:
                        level = 2  # Section

                    sections.append(PDFSection(
                        title=line_text,
                        content="",  # Will be filled later
                        level=level,
                        page_start=page_num,
                        page_end=page_num
                    ))

        except Exception as e:
            # If structure detection fails, just skip it
            print(f"Warning: Could not detect structure on page {page_num}: {e}")
            return []

        return sections

    def _group_words_into_lines(self, words: List[dict]) -> List[dict]:
        """Group words into lines based on vertical position."""
        if not words:
            return []

        lines = []
        current_line = {'words': [], 'text': '', 'top': None}

        for word in words:
            word_top = word.get('top', 0)

            if current_line['top'] is None:
                current_line['top'] = word_top

            # If word is on same line (within 2 points)
            if abs(word_top - current_line['top']) < 2:
                current_line['words'].append(word)
                current_line['text'] += ' ' + word.get('text', '')
            else:
                # New line
                if current_line['words']:
                    lines.append(current_line)
                current_line = {
                    'words': [word],
                    'text': word.get('text', ''),
                    'top': word_top
                }

        # Add last line
        if current_line['words']:
            lines.append(current_line)

        return lines

    def chunk_by_structure(self, doc: PDFDocument) -> List[PDFChunk]:
        """
        Chunk document based on its structure (sections).

        Args:
            doc: PDFDocument with detected structure

        Returns:
            List of PDFChunks based on sections
        """
        if not doc.has_structure or not doc.sections:
            # Fallback to fixed-size chunking
            return self.chunk_by_size(doc)

        chunks = []
        text_lines = doc.text.split('\n')
        current_section_idx = 0

        for section in doc.sections:
            # Find section content (text between this heading and next)
            section_text = self._extract_section_text(doc.text, section)

            # If section is too long, split it into sub-chunks
            if self._count_words(section_text) > self.chunk_size * 1.5:
                sub_chunks = self._split_into_chunks(section_text, self.chunk_size)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append(PDFChunk(
                        content=sub_chunk,
                        chunk_index=len(chunks),
                        section_title=f"{section.title} (Part {i+1})",
                        page_start=section.page_start,
                        page_end=section.page_end
                    ))
            else:
                chunks.append(PDFChunk(
                    content=section_text,
                    chunk_index=len(chunks),
                    section_title=section.title,
                    page_start=section.page_start,
                    page_end=section.page_end
                ))

        return chunks

    def chunk_by_size(self, doc: PDFDocument) -> List[PDFChunk]:
        """
        Chunk document into fixed-size chunks.

        Args:
            doc: PDFDocument to chunk

        Returns:
            List of fixed-size PDFChunks
        """
        chunks = self._split_into_chunks(doc.text, self.chunk_size)

        return [
            PDFChunk(
                content=chunk,
                chunk_index=i,
                section_title=None,
                page_start=0,
                page_end=doc.num_pages
            )
            for i, chunk in enumerate(chunks)
        ]

    def _extract_section_text(self, full_text: str, section: PDFSection) -> str:
        """Extract text belonging to a specific section."""
        # Simple approach: find section title in text and extract until next section
        # This is a basic implementation - can be improved

        lines = full_text.split('\n')
        section_lines = []
        in_section = False

        for line in lines:
            if section.title in line:
                in_section = True
                continue

            if in_section:
                # Check if we hit another heading
                if line.strip() and len(line.split()) < 15:
                    # Might be next heading - stop here
                    if line.strip()[0].isupper():
                        break

                section_lines.append(line)

        return '\n'.join(section_lines)

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size words."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def get_chunking_strategy(self, doc: PDFDocument) -> str:
        """
        Determine which chunking strategy to use.

        Returns:
            'structure' or 'size'
        """
        if doc.has_structure and len(doc.sections) >= 3:
            return 'structure'
        return 'size'


def create_pdf_processor(chunk_size: int = 1000) -> PDFProcessor:
    """
    Factory function to create a PDFProcessor.

    Args:
        chunk_size: Target number of words per chunk

    Returns:
        Configured PDFProcessor instance
    """
    return PDFProcessor(chunk_size=chunk_size)
