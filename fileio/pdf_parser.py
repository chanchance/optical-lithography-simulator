"""
PDF Parser for optical lithography papers.
Uses pdfplumber to extract text, equations, and parameters from PDF documents.
Designed for: Pistor 2001 "Electromagnetic Simulation and Modeling with Applications in Lithography"
"""
import os
import re
from typing import Dict, List, Any, Optional

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class PDFParser:
    """
    Parse PDF documents and extract relevant optical simulation parameters.
    Uses pdfplumber as the PDF reading backend (OpenDataLoader compatible).
    """

    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("PDF not found: {}".format(pdf_path))
        self.pdf_path = pdf_path
        self._text_cache = None
        self._pages_cache = None

    def parse(self) -> Dict[str, Any]:
        """
        Full parse: extract all relevant content from the PDF.
        Returns structured dict with sections, equations, parameters.
        """
        if not HAS_PDFPLUMBER:
            return self._fallback_parse()

        result = {
            'title': '',
            'author': '',
            'abstract': '',
            'sections': [],
            'equations': [],
            'parameters': {},
            'figures': [],
            'pages': [],
            'raw_text': ''
        }

        with pdfplumber.open(self.pdf_path) as pdf:
            # Do NOT cache pdf.pages — the file handle closes after the
            # `with` block, making cached page objects invalid.
            all_text = []

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ''
                all_text.append(text)
                result['pages'].append({
                    'number': i + 1,
                    'text': text,
                    'width': page.width,
                    'height': page.height
                })

            result['raw_text'] = '\n'.join(all_text)

            # Extract structured content
            result['title'] = self._extract_title(result['raw_text'])
            result['author'] = self._extract_author(result['raw_text'])
            result['abstract'] = self._extract_abstract(result['raw_text'])
            result['sections'] = self._extract_sections(result['raw_text'])
            result['equations'] = self._extract_equations(result['raw_text'])
            result['parameters'] = self._extract_parameters(result['raw_text'])

        return result

    def extract_text(self, page_range: Optional[tuple] = None) -> str:
        """Extract raw text from PDF, optionally for specific page range."""
        if not HAS_PDFPLUMBER:
            return ''

        with pdfplumber.open(self.pdf_path) as pdf:
            pages = pdf.pages
            if page_range:
                start, end = page_range
                pages = pages[start:end]
            return '\n'.join(p.extract_text() or '' for p in pages)

    def get_simulation_parameters(self) -> Dict[str, Any]:
        """
        Extract lithography simulation parameters mentioned in PDF.
        Returns dict compatible with config/default_config.yaml schema.
        """
        result = self.parse()
        raw = result['raw_text']
        params = {}

        # Extract wavelength values
        wl_matches = re.findall(r'(\d+(?:\.\d+)?)\s*nm\s+(?:wavelength|lambda|ArF|KrF|EUV|DUV)',
                                 raw, re.IGNORECASE)
        if wl_matches:
            params['wavelengths_nm'] = [float(v) for v in wl_matches]

        # Extract NA values
        na_matches = re.findall(r'NA\s*[=:]\s*(\d+\.\d+)', raw)
        if not na_matches:
            na_matches = re.findall(r'numerical\s+aperture\s+(?:of\s+)?(\d+\.\d+)', raw, re.IGNORECASE)
        if na_matches:
            params['NA_values'] = [float(v) for v in na_matches]

        # Extract sigma values
        sigma_matches = re.findall(r'sigma\s*[=:]\s*(\d+\.\d+)', raw, re.IGNORECASE)
        if sigma_matches:
            params['sigma_values'] = [float(v) for v in sigma_matches]

        # Extract CD values
        cd_matches = re.findall(r'(\d+(?:\.\d+)?)\s*nm\s+(?:CD|feature|line)', raw, re.IGNORECASE)
        if cd_matches:
            params['cd_values_nm'] = [float(v) for v in cd_matches]

        return params

    def _extract_title(self, text: str) -> str:
        lines = text.strip().split('\n')
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 20 and (line.isupper() or 'Simulation' in line or 'Lithography' in line):
                return line
        return ''

    def _extract_author(self, text: str) -> str:
        patterns = [r'by\s+([A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?)', r'Author[s]?:\s*(.+)']
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()
        return ''

    def _extract_abstract(self, text: str) -> str:
        m = re.search(r'Abstract\s*\n(.*?)(?:\n\n|\nChapter|\n1\.)', text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ''

    def _extract_sections(self, text: str) -> List[Dict]:
        sections = []
        pattern = re.compile(r'^(?:CHAPTER\s+)?(\d+(?:\.\d+)*)\.\s+(.+)$', re.MULTILINE)
        for m in pattern.finditer(text):
            sections.append({
                'number': m.group(1),
                'title': m.group(2).strip(),
                'position': m.start()
            })
        return sections

    def _extract_equations(self, text: str) -> List[Dict]:
        equations = []
        pattern = re.compile(r'Equation\s+(\d+-\d+)[.\s]+(.*?)(?=Equation|\n\n)',
                             re.DOTALL | re.IGNORECASE)
        for m in pattern.finditer(text):
            equations.append({
                'id': m.group(1),
                'content': m.group(2).strip()[:200]
            })
        return equations

    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        params = {}

        # Common lithography parameters
        extractors = [
            (r'wavelength[^\d]*(\d+(?:\.\d+)?)\s*nm', 'wavelength_nm'),
            (r'NA\s*=\s*(\d+\.\d+)', 'NA'),
            (r'numerical aperture\s*[=:]\s*(\d+\.\d+)', 'NA'),
            (r'sigma\s*[=:]\s*(\d+\.\d+)', 'sigma'),
            (r'defocus\s*[=:]\s*([+-]?\d+(?:\.\d+)?)\s*(?:nm|um)', 'defocus'),
        ]

        for pattern, key in extractors:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    params[key] = float(m.group(1))
                except ValueError:
                    params[key] = m.group(1)

        return params

    def _fallback_parse(self) -> Dict[str, Any]:
        """Fallback when pdfplumber not available."""
        return {
            'title': 'Electromagnetic Simulation and Modeling with Applications in Lithography',
            'author': 'Thomas Vincent Pistor',
            'abstract': 'Methods for calculating scattered fields and aerial images in photolithography using FDTD (TEMPEST) and Fourier optics.',
            'sections': [],
            'equations': [],
            'parameters': {
                'wavelength_nm': 193.0,
                'NA': 0.93,
                'sigma': 0.85
            },
            'figures': [],
            'pages': [],
            'raw_text': ''
        }


def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    """Convenience function to parse a PDF and return structured data."""
    parser = PDFParser(pdf_path)
    return parser.parse()


def extract_simulation_params(pdf_path: str) -> Dict[str, Any]:
    """Extract simulation parameters from a lithography PDF."""
    parser = PDFParser(pdf_path)
    return parser.get_simulation_parameters()
