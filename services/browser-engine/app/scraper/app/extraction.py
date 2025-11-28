"""Content extraction helpers."""

from __future__ import annotations

import io
import mimetypes
from typing import Any, Dict, List

import PyPDF2
import trafilatura
from bs4 import BeautifulSoup

from .logging import get_logger
from .models import ContentType

logger = get_logger()


class ContentExtractor:
    """Extract and clean content using multiple parsing strategies."""

    def __init__(self) -> None:
        self.extraction_methods = [
            self._extract_with_trafilatura,
            self._extract_with_beautifulsoup_smart,
            self._extract_with_beautifulsoup_fallback,
        ]

    def extract_content(
        self, content: bytes, url: str, content_type: str = ""
    ) -> Dict[str, Any]:
        """Extract clean content from various content types."""
        try:
            detected_type = self._detect_content_type(content, content_type, url)

            if detected_type == ContentType.PDF:
                return self._extract_pdf_content(content, url)
            if detected_type == ContentType.HTML:
                return self._extract_html_content(
                    content.decode("utf-8", errors="ignore"), url
                )
            if detected_type == ContentType.TEXT:
                return self._extract_text_content(
                    content.decode("utf-8", errors="ignore"), url
                )

            return self._extract_html_content(
                content.decode("utf-8", errors="ignore"), url
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Content extraction failed", url=url, error=str(exc))
            return self._empty_result(url)

    def _detect_content_type(
        self, content: bytes, content_type: str, url: str
    ) -> ContentType:
        """Detect content type from headers, content, and URL."""
        if content_type:
            lowered = content_type.lower()
            if "pdf" in lowered:
                return ContentType.PDF
            if "html" in lowered:
                return ContentType.HTML
            if "text" in lowered:
                return ContentType.TEXT

        guessed_type, _ = mimetypes.guess_type(url)
        if guessed_type:
            if "pdf" in guessed_type:
                return ContentType.PDF
            if "html" in guessed_type:
                return ContentType.HTML
            if guessed_type.startswith("text/"):
                return ContentType.TEXT

        if url.lower().endswith(".pdf") or content.startswith(b"%PDF"):
            return ContentType.PDF
        if b"<html" in content[:1000].lower() or b"<!doctype html" in content[:1000].lower():
            return ContentType.HTML

        return ContentType.HTML

    def _extract_html_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content from HTML using multiple strategies."""
        best_result: Dict[str, Any] | None = None
        best_score = 0.0
        extraction_method = "none"

        for method in self.extraction_methods:
            try:
                result = method(html, url)
                score = self._calculate_quality_score(result)
                if score > best_score:
                    best_result = result
                    best_score = score
                    extraction_method = method.__name__
                if score > 0.8:
                    break
            except Exception as exc:
                logger.debug(
                    "Extraction method failed",
                    method=method.__name__,
                    url=url,
                    error=str(exc),
                )

        if best_result is None:
            return self._empty_result(url)

        best_result["quality_score"] = best_score
        best_result["extraction_method"] = extraction_method
        best_result["content_type"] = ContentType.HTML
        return best_result

    def _extract_with_trafilatura(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using trafilatura."""
        content = trafilatura.extract(
            html, include_comments=False, include_tables=True, include_formatting=True
        )
        if not content or len(content.strip()) < 20:
            raise ValueError("Trafilatura extraction failed or insufficient content")

        soup = BeautifulSoup(html, "html.parser")
        title = self._extract_title(soup)

        return {
            "title": title,
            "content": content.strip(),
            "word_count": len(content.split()),
            "quality_score": 0.0,
            "extraction_method": "_extract_with_trafilatura",
            "content_type": ContentType.HTML,
        }

    def _extract_with_beautifulsoup_smart(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using BeautifulSoup with smart heuristics."""
        soup = BeautifulSoup(html, "html.parser")
        self._remove_noise(soup)

        article = soup.find("article")
        if not article:
            article = soup.find("main")
        if not article:
            article = soup.body

        if article:
            for selector in ["script", "style", "nav", "footer", "header", "aside"]:
                for element in article.select(selector):
                    element.decompose()

            text_blocks: List[str] = []

            for heading in article.find_all(["h1", "h2", "h3"]):
                heading_text = heading.get_text(strip=True)
                if heading_text:
                    text_blocks.append(heading_text)

            for paragraph in article.find_all("p"):
                paragraph_text = paragraph.get_text(strip=True)
                if paragraph_text:
                    text_blocks.append(paragraph_text)

            content = "\n".join(text_blocks)
        else:
            content = soup.get_text(" ", strip=True)

        title = self._extract_title(soup)
        return {
            "title": title,
            "content": content,
            "word_count": len(content.split()),
            "quality_score": 0.0,
            "extraction_method": "_extract_with_beautifulsoup_smart",
            "content_type": ContentType.HTML,
        }

    def _extract_with_beautifulsoup_fallback(self, html: str, url: str) -> Dict[str, Any]:
        """Fallback extraction when other methods fail."""
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        title = self._extract_title(soup)

        return {
            "title": title,
            "content": text,
            "word_count": len(text.split()),
            "quality_score": 0.0,
            "extraction_method": "_extract_with_beautifulsoup_fallback",
            "content_type": ContentType.HTML,
        }

    def _extract_pdf_content(self, content: bytes, url: str) -> Dict[str, Any]:
        """Extract content from PDF files."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"

            text_content = " ".join(text_content.split())

            title = ""
            if pdf_reader.metadata and pdf_reader.metadata.get("/Title"):
                title = str(pdf_reader.metadata["/Title"])
            else:
                first_line = text_content.split("\n")[0] if text_content else ""
                if len(first_line) < 100:
                    title = first_line.strip()

            return {
                "title": title,
                "content": text_content,
                "word_count": len(text_content.split()) if text_content else 0,
                "quality_score": 0.8 if text_content else 0.0,
                "extraction_method": "_extract_pdf_content",
                "content_type": ContentType.PDF,
            }
        except Exception as exc:
            logger.error("PDF extraction failed", url=url, error=str(exc))
            return self._empty_result(url)

    def _extract_text_content(self, text: str, url: str) -> Dict[str, Any]:
        """Extract content from plain text files."""
        lines = text.split("\n")
        title = lines[0].strip() if lines else ""
        if len(title) > 100:
            title = ""

        return {
            "title": title,
            "content": text.strip(),
            "word_count": len(text.split()) if text else 0,
            "quality_score": 0.7 if text.strip() else 0.0,
            "extraction_method": "_extract_text_content",
            "content_type": ContentType.TEXT,
        }

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_sources = [
            lambda: soup.find("title"),
            lambda: soup.find("h1"),
            lambda: soup.find("meta", property="og:title"),
            lambda: soup.find("meta", attrs={"name": "title"}),
        ]

        for source in title_sources:
            try:
                element = source()
                if element:
                    if getattr(element, "name", "") == "meta":
                        title = element.get("content", "")
                    else:
                        title = element.get_text(strip=True)
                    if title and len(title.strip()) > 0:
                        return title.strip()
            except Exception:
                continue
        return ""

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """Remove script/style tags and other noisy elements."""
        for selector in ["script", "style", "noscript", "iframe", "svg"]:
            for element in soup.select(selector):
                element.decompose()

    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score for extracted content."""
        content = result.get("content", "")
        title = result.get("title", "")
        word_count = result.get("word_count", 0)

        score = 0.0

        if word_count > 500:
            score += 0.4
        elif word_count > 100:
            score += 0.3
        elif word_count > 50:
            score += 0.2
        elif word_count > 10:
            score += 0.1

        if title and len(title.strip()) > 0:
            score += 0.2

        if content:
            if any(
                indicator in content.lower()
                for indicator in ["paragraph", "section", "chapter"]
            ):
                score += 0.1

            meaningful_words = [
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
            ]
            word_list = content.lower().split()
            meaningful_ratio = sum(
                1 for word in word_list if word in meaningful_words
            ) / max(len(word_list), 1)

            if meaningful_ratio > 0.1:
                score += 0.2
            elif meaningful_ratio > 0.05:
                score += 0.1

            unique_words = len(set(word_list))
            repetition_ratio = unique_words / max(len(word_list), 1)
            if repetition_ratio < 0.3:
                score -= 0.2

        return min(max(score, 0.0), 1.0)

    def _empty_result(self, url: str) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "title": "",
            "content": "",
            "word_count": 0,
            "quality_score": 0.0,
            "extraction_method": "failed",
            "content_type": ContentType.UNKNOWN,
        }


__all__ = ["ContentExtractor", "trafilatura"]
