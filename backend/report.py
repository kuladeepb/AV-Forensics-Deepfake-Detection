"""
report.py  –  AV-Forensics Backend
Generates a professional 2-4 page forensic PDF report using ReportLab.
"""

from __future__ import annotations
import base64, io, json, os, textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String, PolyLine
from reportlab.graphics import renderPDF


# ── Colour palette ─────────────────────────────────────────────────────────────
C_DARK      = colors.HexColor("#0D1B2A")
C_BLUE      = colors.HexColor("#1565C0")
C_RED       = colors.HexColor("#B71C1C")
C_GREEN     = colors.HexColor("#1B5E20")
C_AMBER     = colors.HexColor("#E65100")
C_LIGHT     = colors.HexColor("#E8EAF6")
C_DIVIDER   = colors.HexColor("#90A4AE")
C_LABEL     = colors.HexColor("#455A64")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=22, textColor=C_DARK, spaceAfter=4, leading=26,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=10, textColor=C_LABEL, spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "section", parent=base["Heading2"],
            fontSize=12, textColor=C_BLUE, spaceBefore=14, spaceAfter=6,
            borderPad=2, leading=16, fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=9.5, textColor=C_DARK, leading=14, alignment=TA_JUSTIFY,
        ),
        "label": ParagraphStyle(
            "label", parent=base["Normal"],
            fontSize=8.5, textColor=C_LABEL, leading=12,
        ),
        "verdict_fake": ParagraphStyle(
            "verdict_fake", parent=base["Normal"],
            fontSize=28, textColor=C_RED, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=6, leading=34,
        ),
        "verdict_real": ParagraphStyle(
            "verdict_real", parent=base["Normal"],
            fontSize=28, textColor=C_GREEN, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=6, leading=34,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"],
            fontSize=7.5, textColor=C_LABEL, alignment=TA_CENTER,
        ),
        "warning": ParagraphStyle(
            "warning", parent=base["Normal"],
            fontSize=9, textColor=C_AMBER, leading=13, alignment=TA_JUSTIFY,
        ),
    }


def _confidence_bar(confidence: float, is_fake: bool, width: float = 14 * cm) -> Drawing:
    """Draw a horizontal confidence bar."""
    h = 18
    d = Drawing(width, h + 10)
    # background
    d.add(Rect(0, 5, width, h, fillColor=C_LIGHT, strokeColor=None))
    # fill
    fill_w = width * confidence
    fill_c = C_RED if is_fake else C_GREEN
    d.add(Rect(0, 5, fill_w, h, fillColor=fill_c, strokeColor=None))
    # label
    d.add(String(fill_w + 4, 8, f"{confidence * 100:.1f}%", fontSize=9, fillColor=C_DARK))
    return d


def _sync_chart(
    sync_timeline: List[dict],
    suspects: List[dict],
    width: float = 14 * cm,
    height: float = 4 * cm,
) -> Drawing:
    """Draw a mini sync-score timeline chart."""
    if not sync_timeline:
        return Drawing(width, height)

    d  = Drawing(width, height)
    pw = width  - 20
    ph = height - 20
    ox, oy = 10, 10

    # Background
    d.add(Rect(ox, oy, pw, ph, fillColor=C_LIGHT, strokeColor=C_DIVIDER, strokeWidth=0.5))

    # Shade suspect regions
    t_max = sync_timeline[-1]["t"] or 1.0
    for seg in suspects:
        x1 = ox + (seg["start"] / t_max) * pw
        x2 = ox + (seg["end"]   / t_max) * pw
        d.add(Rect(x1, oy, x2 - x1, ph, fillColor=colors.HexColor("#FFCDD2"), strokeColor=None))

    # Threshold line
    ty = oy + 0.5 * ph
    d.add(PolyLine([ox, ty, ox + pw, ty], strokeColor=C_AMBER, strokeWidth=0.6, strokeDashArray=[3, 3]))

    # Score line
    pts: List[float] = []
    for entry in sync_timeline:
        x = ox + (entry["t"]     / t_max) * pw
        y = oy + entry["score"] * ph
        pts += [x, y]
    if len(pts) >= 4:
        d.add(PolyLine(pts, strokeColor=C_BLUE, strokeWidth=1.2))

    return d


def _b64_to_image(b64str: str, width_cm: float, height_cm: float) -> Optional[Image]:
    try:
        data = base64.b64decode(b64str)
        buf  = io.BytesIO(data)
        return Image(buf, width=width_cm * cm, height=height_cm * cm)
    except Exception:
        return None


# ── Page header / footer callbacks ────────────────────────────────────────────

def _make_page_template(doc):
    MARGIN = 2 * cm
    frame  = Frame(MARGIN, MARGIN + 1.5 * cm, doc.width, doc.height - 1.5 * cm, id="main")
    return PageTemplate(id="standard", frames=[frame], onPage=_draw_header_footer)


def _draw_header_footer(canvas, doc):
    canvas.saveState()
    W, H = A4
    # Top rule
    canvas.setStrokeColor(C_BLUE)
    canvas.setLineWidth(1.5)
    canvas.line(2 * cm, H - 1.4 * cm, W - 2 * cm, H - 1.4 * cm)
    # Header text
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_LABEL)
    canvas.drawString(2 * cm, H - 1.1 * cm, "AV-FORENSICS  |  Audio-Visual Deepfake Detection System")
    canvas.drawRightString(W - 2 * cm, H - 1.1 * cm, f"CONFIDENTIAL — Page {doc.page}")
    # Footer rule
    canvas.setStrokeColor(C_DIVIDER)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.5 * cm, W - 2 * cm, 1.5 * cm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_LABEL)
    canvas.drawCentredString(W / 2, 0.9 * cm,
        "This report is generated by an automated AI system. Results are probabilistic and NOT legal proof.")
    canvas.restoreState()


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_report(result: dict, case_id: str, output_path: str) -> str:
    """
    Generate a PDF forensic report and save to output_path.
    Returns output_path.
    """
    st = _styles()
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    is_fake = result["verdict"] == "FAKE"
    verdict_color = C_RED if is_fake else C_GREEN

    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )
    doc.addPageTemplates([_make_page_template(doc)])
    story = []

    # ── Page 1: Header & Verdict ────────────────────────────────────────────
    story.append(Paragraph("FORENSIC ANALYSIS REPORT", st["title"]))
    story.append(Paragraph("Audio-Visual Lip-Sync Deepfake Detection", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_DIVIDER))
    story.append(Spacer(1, 0.4 * cm))

    # Case info table
    modality = result.get("modality", {})
    meta = [
        ["Case ID",         case_id],
        ["Date / Time",     now],
        ["File Name",       result.get("filename", "—")],
        ["SHA-256",         result.get("sha256", "—")[:32] + "…"],
        ["Duration",        f"{result.get('duration', '—')} s"],
    ]
    t = Table(meta, colWidths=[4 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1),  C_LABEL),
        ("TEXTCOLOR",   (1, 0), (1, -1),  C_DARK),
        ("FONTNAME",    (0, 0), (0, -1),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, C_LIGHT]),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6 * cm))

    # Verdict
    story.append(Paragraph("VERDICT", st["section"]))
    verdict_style = st["verdict_fake"] if is_fake else st["verdict_real"]
    story.append(Paragraph(result["verdict"], verdict_style))
    story.append(Paragraph(f"Confidence: {result.get('confidence', 0) * 100:.1f}%", st["body"]))
    story.append(_confidence_bar(result.get("confidence", 0), is_fake))
    story.append(Spacer(1, 0.4 * cm))

    # Modality finding
    story.append(Paragraph("MODALITY FINDING", st["section"]))
    mod_label = modality.get("label", "—")
    mod_class = modality.get("class", "—")
    story.append(Paragraph(
        f"<b>Detected class:</b> {mod_class} — <b>{mod_label}</b>",
        st["body"],
    ))
    if is_fake:
        af = modality.get("audio_fake_prob", 0)
        vf = modality.get("video_fake_prob", 0)
        story.append(Paragraph(
            f"Audio manipulation probability: <b>{af * 100:.1f}%</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Video manipulation probability: <b>{vf * 100:.1f}%</b>",
            st["body"],
        ))

    prob_data = [["Class", "Label", "Probability"]] + [
        [cls, _class_label(cls), f"{prob * 100:.2f}%"]
        for cls, prob in modality.get("probabilities", {}).items()
    ]
    pt = Table(prob_data, colWidths=[3 * cm, 7 * cm, 5 * cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  C_BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_LIGHT]),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(pt)
    story.append(Spacer(1, 0.5 * cm))

    # ── Page 2: Sync Timeline + Evidence Frames ─────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("LIP-AUDIO SYNC TIMELINE", st["section"]))
    sync_tl  = result.get("sync_timeline", [])
    suspects = result.get("suspect_segments", [])

    story.append(Paragraph(
        "The chart shows per-frame cosine similarity between lip and audio embeddings. "
        "Low scores (below 0.5 dashed line) indicate potential desynchronisation. "
        "Red-shaded regions are suspect segments.",
        st["body"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_sync_chart(sync_tl, suspects, width=14 * cm, height=4 * cm))

    if suspects:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Suspect Segments:", st["label"]))
        seg_data = [["Start (s)", "End (s)", "Mean Sync Score"]] + [
            [f"{s['start']:.2f}", f"{s['end']:.2f}", f"{s['mean_score']:.4f}"]
            for s in suspects
        ]
        st2 = Table(seg_data, colWidths=[4 * cm, 4 * cm, 7 * cm])
        st2.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), C_AMBER),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ]))
        story.append(st2)

    story.append(Spacer(1, 0.5 * cm))

    # Evidence Frames
    story.append(Paragraph("EVIDENCE FRAMES", st["section"]))
    story.append(Paragraph(
        "Frames with lowest lip-audio sync scores. Each pair shows the full frame "
        "and the extracted mouth region. Timestamps are indicative "
        "(the model has no frame-level ground-truth labels).",
        st["body"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    ev_frames = result.get("evidence_frames", [])
    for i in range(0, len(ev_frames), 2):
        row_imgs  = []
        row_cells = []
        for j in range(2):
            if i + j >= len(ev_frames):
                row_cells.append("")
                continue
            ef = ev_frames[i + j]
            cell_parts = []
            full_img  = _b64_to_image(ef.get("frame_b64", ""), 3.2, 2.4)
            mouth_img = _b64_to_image(ef.get("mouth_b64", ""), 1.4, 1.4)
            if full_img:
                cell_parts.append(full_img)
            if mouth_img:
                cell_parts.append(mouth_img)
            cell_parts.append(Paragraph(
                f"t = {ef['t']:.2f}s  |  sync = {ef['score']:.4f}", st["label"]
            ))
            row_cells.append(cell_parts)

        ev_row = Table([row_cells], colWidths=[7 * cm, 7 * cm])
        ev_row.setStyle(TableStyle([
            ("VALIGN",  (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ev_row)
        story.append(Spacer(1, 0.2 * cm))

    # ── Page 3: Methodology + Limitations ───────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("METHODOLOGY", st["section"]))
    story.append(Paragraph(
        "<b>Visual Encoder:</b> Lip Video Transformer — mouth crops extracted with MediaPipe "
        "face mesh, 96×96 grayscale, 32 frames per 4-second window, encoded with a "
        "spatial + temporal transformer (patch size 16, 256-dim embeddings).<br/><br/>"
        f"<b>Audio Encoder:</b> {result.get('_audio_model', 'utter-project/mHuBERT-147')} — "
        "CNN feature extractor frozen; lower transformer layers frozen during training; "
        "learnable weighted sum over all hidden layers; projected to 256 dimensions.<br/><br/>"
        "<b>Fusion:</b> Cross-modal transformer over concatenated visual and audio tokens "
        "(2 layers, 4 heads), mean-pooled to a single fused representation.<br/><br/>"
        "<b>Inference:</b> Sliding-window (4 s window, 2 s stride) over the uploaded video. "
        "Window predictions are averaged. Sync timeline = per-time-step cosine similarity "
        "between pre-fusion visual and audio embeddings.<br/><br/>"
        "<b>Training dataset:</b> FakeAVCeleb (Kaggle: shreyaty08/fakeavceleb). "
        "Classes: RVRA (real video + real audio), RVFA (real video + fake audio), "
        "FVRA (fake video + real audio), FVFA (fake video + fake audio).<br/><br/>"
        f"<b>Model version:</b> {result.get('_model_version', '1.0.0')}",
        st["body"],
    ))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("LIMITATIONS", st["section"]))
    story.append(Paragraph(
        "⚠ <b>Important — please read before acting on this report:</b>",
        st["warning"],
    ))
    limitations = [
        "Results are <b>probabilistic</b>. The model outputs confidence scores, not certainties. "
        "A high-confidence result can still be wrong.",
        "The model was trained solely on <b>FakeAVCeleb</b>, which contains English-language "
        "speech from a specific set of celebrity videos. Performance on other languages, "
        "accents, or recording conditions has not been validated.",
        "Multilingual deepfakes, non-speech audio tracks, or heavily compressed videos may "
        "degrade detection accuracy.",
        "Evidence frame timestamps are <b>indicative only</b>. The training data contains no "
        "frame-level annotations, so the sync-score timeline identifies suspicious regions "
        "but cannot localise the exact manipulated frames.",
        "This report is generated by an <b>automated AI system</b>. It does not constitute "
        "legal proof of any kind and should not be used as the sole basis for legal, "
        "financial, or editorial decisions.",
        "The system does not store uploaded videos after processing. Results are cached "
        "in memory for approximately one hour.",
    ]
    for lim in limitations:
        story.append(Paragraph(f"• {lim}", st["body"]))
        story.append(Spacer(1, 0.15 * cm))

    doc.build(story)
    return output_path


def _class_label(cls: str) -> str:
    return {
        "RVRA": "Authentic",
        "RVFA": "Audio Manipulated",
        "FVRA": "Video Manipulated",
        "FVFA": "Both Manipulated",
    }.get(cls, cls)

