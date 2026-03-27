# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""PDF rendering for AI Act readiness bundles."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen.canvas import Canvas

type MarkdownLineKind = Literal["heading1", "heading2", "bullet", "note", "body"]

_PAGE_WIDTH: float = float(A4[0])
_PAGE_HEIGHT: float = float(A4[1])
_LEFT_MARGIN: float = 56.0
_RIGHT_MARGIN: float = 56.0
_TOP_MARGIN: float = 72.0
_BOTTOM_MARGIN: float = 54.0
_HEADER_GAP: float = 26.0
_FOOTER_GAP: float = 24.0
_BODY_WIDTH: float = _PAGE_WIDTH - _LEFT_MARGIN - _RIGHT_MARGIN
_TOP_CONTENT_Y: float = _PAGE_HEIGHT - _TOP_MARGIN - _HEADER_GAP
_MIN_CONTENT_Y: float = _BOTTOM_MARGIN + _FOOTER_GAP


@dataclass(frozen=True, slots=True)
class _TextStyle:
    """Simple text style for low-level PDF drawing."""

    font_name: str
    font_size: float
    leading: float
    color_hex: str
    top_space: float
    bottom_space: float
    left_indent: float = 0.0
    bullet_indent: float = 0.0


@dataclass(slots=True)
class _PDFState:
    """Mutable drawing state for the current PDF page."""

    canvas: Canvas
    company_name: str
    system_name: str
    current_y: float = _TOP_CONTENT_Y


def render_ai_act_report_pdf(
    *,
    company_name: str,
    system_name: str,
    generated_at: str,
    overall_status: str,
    risk_level: str,
    bundle_fingerprint: str,
    executive_summary_markdown: str,
    remediation_markdown: str,
    auditor_appendix_markdown: str,
    fria_prep_markdown: str | None,
) -> bytes:
    """Render a deterministic AI Act report PDF."""
    buffer = BytesIO()
    canvas = Canvas(
        buffer,
        pagesize=A4,
        invariant=1,
        pageCompression=0,
    )
    canvas.setCreator("Vauban")
    canvas.setAuthor("Vauban")
    canvas.setTitle(f"AI Act readiness report - {company_name}")
    canvas.setSubject(f"AI Act readiness report for {system_name}")

    state = _PDFState(
        canvas=canvas,
        company_name=company_name,
        system_name=system_name,
    )
    _draw_page_chrome(state)
    _draw_cover_page(
        state,
        generated_at=generated_at,
        overall_status=overall_status,
        risk_level=risk_level,
        bundle_fingerprint=bundle_fingerprint,
    )

    sections: list[tuple[str, str]] = [
        ("Executive Report", executive_summary_markdown),
        ("Remediation Plan", remediation_markdown),
    ]
    if fria_prep_markdown is not None:
        sections.append(("FRIA Preparation Pack", fria_prep_markdown))
    sections.append(("Auditor Appendix", auditor_appendix_markdown))

    for title, markdown in sections:
        _new_page(state)
        _render_markdown_document(
            state,
            title=title,
            markdown=markdown,
        )

    canvas.save()
    return buffer.getvalue()


def _draw_page_chrome(state: _PDFState) -> None:
    """Draw the page header and footer."""
    page_number = state.canvas.getPageNumber()
    state.canvas.saveState()
    state.canvas.setStrokeColor(HexColor("#D7DEE8"))
    state.canvas.setFillColor(HexColor("#213547"))
    state.canvas.setLineWidth(0.6)
    state.canvas.setFont("Helvetica-Bold", 10)
    state.canvas.drawString(
        _LEFT_MARGIN,
        _PAGE_HEIGHT - _TOP_MARGIN + 8.0,
        "Vauban AI Act readiness report",
    )
    state.canvas.setFont("Helvetica", 9)
    state.canvas.drawRightString(
        _PAGE_WIDTH - _RIGHT_MARGIN,
        _PAGE_HEIGHT - _TOP_MARGIN + 8.0,
        f"{state.company_name} / {state.system_name}",
    )
    state.canvas.line(
        _LEFT_MARGIN,
        _PAGE_HEIGHT - _TOP_MARGIN,
        _PAGE_WIDTH - _RIGHT_MARGIN,
        _PAGE_HEIGHT - _TOP_MARGIN,
    )
    state.canvas.line(
        _LEFT_MARGIN,
        _BOTTOM_MARGIN,
        _PAGE_WIDTH - _RIGHT_MARGIN,
        _BOTTOM_MARGIN,
    )
    state.canvas.setFillColor(HexColor("#5C6F82"))
    state.canvas.setFont("Helvetica", 8)
    state.canvas.drawString(
        _LEFT_MARGIN,
        _BOTTOM_MARGIN - 12.0,
        "Prepared by Vauban. Readiness evidence only; not legal certification.",
    )
    state.canvas.drawRightString(
        _PAGE_WIDTH - _RIGHT_MARGIN,
        _BOTTOM_MARGIN - 12.0,
        f"Page {page_number}",
    )
    state.canvas.restoreState()
    state.current_y = _TOP_CONTENT_Y


def _new_page(state: _PDFState) -> None:
    """Advance to a new page and redraw page chrome."""
    state.canvas.showPage()
    _draw_page_chrome(state)


def _draw_cover_page(
    state: _PDFState,
    *,
    generated_at: str,
    overall_status: str,
    risk_level: str,
    bundle_fingerprint: str,
) -> None:
    """Render the report cover page."""
    _draw_paragraph(
        state,
        "AI Act Readiness Report",
        _style_for("heading1"),
    )
    _draw_paragraph(
        state,
        (
            "Combined executive and auditor report pack for deployer-facing"
            " AI Act readiness review."
        ),
        _style_for("body"),
    )
    _draw_paragraph(
        state,
        f"Company: {state.company_name}",
        _style_for("heading2"),
    )
    _draw_paragraph(
        state,
        f"System: {state.system_name}",
        _style_for("note"),
    )
    _draw_paragraph(
        state,
        f"Generated at: {generated_at}",
        _style_for("note"),
    )
    _draw_bullet(
        state,
        f"Overall status: {overall_status}",
    )
    _draw_bullet(
        state,
        f"Risk level: {risk_level}",
    )
    _draw_bullet(
        state,
        f"Bundle fingerprint: {bundle_fingerprint}",
    )
    _draw_paragraph(
        state,
        (
            "This PDF is derived from the same evidence bundle as the JSON"
            " and Markdown outputs. It summarizes readiness findings and"
            " reviewer-facing appendix material in one document."
        ),
        _style_for("body"),
    )


def _render_markdown_document(
    state: _PDFState,
    *,
    title: str,
    markdown: str,
) -> None:
    """Render one Markdown document into the current PDF."""
    _draw_paragraph(state, title, _style_for("heading2"))
    paragraph_lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            _flush_paragraph(state, paragraph_lines)
            state.current_y -= 4.0
            continue
        if stripped.startswith("# "):
            _flush_paragraph(state, paragraph_lines)
            _draw_paragraph(
                state,
                stripped[2:].strip(),
                _style_for("heading1"),
            )
            continue
        if stripped.startswith("## "):
            _flush_paragraph(state, paragraph_lines)
            _draw_paragraph(
                state,
                stripped[3:].strip(),
                _style_for("heading2"),
            )
            continue
        if stripped.startswith("- "):
            _flush_paragraph(state, paragraph_lines)
            _draw_bullet(
                state,
                stripped[2:].strip(),
            )
            continue
        if raw_line.startswith("  "):
            _flush_paragraph(state, paragraph_lines)
            _draw_paragraph(state, stripped, _style_for("note"))
            continue
        paragraph_lines.append(stripped)
    _flush_paragraph(state, paragraph_lines)


def _flush_paragraph(state: _PDFState, paragraph_lines: list[str]) -> None:
    """Draw and clear a buffered paragraph."""
    if not paragraph_lines:
        return
    _draw_paragraph(
        state,
        " ".join(paragraph_lines),
        _style_for("body"),
    )
    paragraph_lines.clear()


def _style_for(kind: MarkdownLineKind) -> _TextStyle:
    """Return one built-in text style."""
    if kind == "heading1":
        return _TextStyle(
            font_name="Helvetica-Bold",
            font_size=19.0,
            leading=23.0,
            color_hex="#102A43",
            top_space=8.0,
            bottom_space=8.0,
        )
    if kind == "heading2":
        return _TextStyle(
            font_name="Helvetica-Bold",
            font_size=13.0,
            leading=17.0,
            color_hex="#17324D",
            top_space=10.0,
            bottom_space=4.0,
        )
    if kind == "note":
        return _TextStyle(
            font_name="Helvetica",
            font_size=9.0,
            leading=12.0,
            color_hex="#486581",
            top_space=0.0,
            bottom_space=2.0,
            left_indent=18.0,
        )
    return _TextStyle(
        font_name="Helvetica",
        font_size=10.0,
        leading=13.0,
        color_hex="#243B53",
        top_space=0.0,
        bottom_space=4.0,
    )


def _draw_paragraph(
    state: _PDFState,
    text: str,
    style: _TextStyle,
) -> None:
    """Draw wrapped paragraph text with spacing and pagination."""
    lines = simpleSplit(
        text,
        style.font_name,
        style.font_size,
        _BODY_WIDTH - style.left_indent,
    )
    required_height = (
        style.top_space
        + style.bottom_space
        + (max(len(lines), 1) * style.leading)
    )
    _ensure_space(state, required_height)
    state.current_y -= style.top_space
    state.canvas.saveState()
    state.canvas.setFillColor(HexColor(style.color_hex))
    state.canvas.setFont(style.font_name, style.font_size)
    text_x = _LEFT_MARGIN + style.left_indent
    if not lines:
        lines = [text]
    for line in lines:
        state.canvas.drawString(text_x, state.current_y, line)
        state.current_y -= style.leading
    state.canvas.restoreState()
    state.current_y -= style.bottom_space


def _draw_bullet(state: _PDFState, text: str) -> None:
    """Draw a wrapped bullet line."""
    style = _TextStyle(
        font_name="Helvetica",
        font_size=10.0,
        leading=13.0,
        color_hex="#243B53",
        top_space=0.0,
        bottom_space=3.0,
        left_indent=0.0,
        bullet_indent=12.0,
    )
    lines = simpleSplit(
        text,
        style.font_name,
        style.font_size,
        _BODY_WIDTH - style.bullet_indent,
    )
    required_height = (
        style.top_space
        + style.bottom_space
        + (max(len(lines), 1) * style.leading)
    )
    _ensure_space(state, required_height)
    state.current_y -= style.top_space
    state.canvas.saveState()
    state.canvas.setFillColor(HexColor(style.color_hex))
    state.canvas.setFont(style.font_name, style.font_size)
    bullet_x = _LEFT_MARGIN + style.left_indent
    text_x = bullet_x + style.bullet_indent
    if not lines:
        lines = [text]
    for index, line in enumerate(lines):
        if index == 0:
            state.canvas.drawString(bullet_x, state.current_y, "-")
        state.canvas.drawString(text_x, state.current_y, line)
        state.current_y -= style.leading
    state.canvas.restoreState()
    state.current_y -= style.bottom_space


def _ensure_space(state: _PDFState, required_height: float) -> None:
    """Start a new page if the current page lacks vertical space."""
    if state.current_y - required_height < _MIN_CONTENT_Y:
        _new_page(state)
