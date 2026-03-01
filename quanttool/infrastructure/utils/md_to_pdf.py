"""Markdown to PDF converter using pandoc and Chrome headless."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
import tempfile
import os


class MarkdownToPDFConverter:
    """Convert Markdown files to PDF using pandoc + Chrome headless.

    This converter uses a two-step process:
    1. pandoc converts Markdown to HTML
    2. Chrome headless converts HTML to PDF

    This approach ensures good Chinese font support without requiring LaTeX.
    """

    # Default Chrome path on macOS
    CHROME_PATHS = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]

    def __init__(self, chrome_path: Optional[str] = None):
        """Initialize the converter.

        Args:
            chrome_path: Path to Chrome/Chromium executable.
                        If not provided, will auto-detect on macOS.
        """
        self.chrome_path = chrome_path or self._find_chrome()
        self._check_dependencies()

    def _find_chrome(self) -> Optional[str]:
        """Find Chrome executable on the system."""
        for path in self.CHROME_PATHS:
            if os.path.exists(path):
                return path
        return None

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        if not shutil.which("pandoc"):
            raise RuntimeError(
                "pandoc is not installed. Install with: brew install pandoc"
            )

        if not self.chrome_path:
            raise RuntimeError(
                "Chrome/Chromium not found. Please install Google Chrome or "
                "provide the chrome_path parameter."
            )

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        open_after: bool = False,
    ) -> str:
        """Convert a Markdown file to PDF.

        Args:
            input_path: Path to the input Markdown file.
            output_path: Path for the output PDF file.
                        If not provided, uses the same name as input with .pdf extension.
            title: Title for the PDF document.
            open_after: Whether to open the PDF after conversion (macOS only).

        Returns:
            Path to the generated PDF file.

        Raises:
            FileNotFoundError: If the input file doesn't exist.
            RuntimeError: If conversion fails.
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Determine output path
        if output_path:
            output_file = Path(output_path)
        else:
            output_file = input_file.with_suffix(".pdf")

        # Create temp HTML file
        tmp_html_path = tempfile.mktemp(suffix=".html")

        try:
            # Step 1: Convert Markdown to HTML using pandoc
            self._md_to_html(input_file, tmp_html_path, title)

            # Step 2: Convert HTML to PDF using Chrome headless
            self._html_to_pdf(tmp_html_path, output_file)

            # Open the PDF if requested
            if open_after:
                subprocess.run(["open", str(output_file)], check=False)

            return str(output_file)

        finally:
            # Clean up temp file
            if os.path.exists(tmp_html_path):
                os.unlink(tmp_html_path)

    def _md_to_html(
        self, input_file: Path, output_html: str, title: Optional[str] = None
    ) -> None:
        """Convert Markdown to HTML using pandoc."""
        cmd = [
            "pandoc",
            str(input_file),
            "-o",
            output_html,
            "--standalone",
        ]

        if title:
            cmd.extend(["--metadata", f"title={title}"])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"pandoc conversion failed: {result.stderr}")

    def _html_to_pdf(self, input_html: str, output_pdf: Path) -> None:
        """Convert HTML to PDF using Chrome headless."""
        cmd = [
            self.chrome_path,
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={output_pdf}",
            input_html,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Chrome PDF conversion failed: {result.stderr}")

        if not output_pdf.exists():
            raise RuntimeError(f"PDF file was not created: {output_pdf}")

    def convert_string(
        self,
        markdown_content: str,
        output_path: str,
        title: Optional[str] = None,
        open_after: bool = False,
    ) -> str:
        """Convert Markdown string to PDF.

        Args:
            markdown_content: Markdown content as string.
            output_path: Path for the output PDF file.
            title: Title for the PDF document.
            open_after: Whether to open the PDF after conversion.

        Returns:
            Path to the generated PDF file.
        """
        # Write content to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".md", delete=False, mode="w", encoding="utf-8"
        ) as tmp_md:
            tmp_md.write(markdown_content)
            tmp_md_path = tmp_md.name

        try:
            return self.convert(tmp_md_path, output_path, title, open_after)
        finally:
            if os.path.exists(tmp_md_path):
                os.unlink(tmp_md_path)


# Convenience function
def convert_md_to_pdf(
    input_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    open_after: bool = False,
) -> str:
    """Convert a Markdown file to PDF.

    This is a convenience function that creates a converter and calls convert().

    Args:
        input_path: Path to the input Markdown file.
        output_path: Path for the output PDF file.
        title: Title for the PDF document.
        open_after: Whether to open the PDF after conversion.

    Returns:
        Path to the generated PDF file.
    """
    converter = MarkdownToPDFConverter()
    return converter.convert(input_path, output_path, title, open_after)