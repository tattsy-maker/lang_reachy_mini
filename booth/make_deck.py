#!/usr/bin/env python3
"""Build the desk deck for the Maker Faire booth: booth/deck/reachy_tutor_deck.pptx

Eight 16:9 slides, one idea each, readable from a metre away. Meant to loop
on the desk display while the robot is busy (auto-advance every 20 s) and to
be pointed at when a visitor asks how it works. Speaker notes carry the
one-line answer for each slide.

Run:
    python3 -m venv /tmp/deckenv && /tmp/deckenv/bin/pip install python-pptx qrcode pillow
    /tmp/deckenv/bin/python booth/make_deck.py

Assets: booth/deck/primary.jpg (the robot) and booth/deck/group.jpg (the
makers) are photos you supply; they are gitignored, and a slide whose photo
is missing gets a plain placeholder panel instead. The QR code is generated
at build time.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import qrcode
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "deck"
OUT = OUT_DIR / "reachy_tutor_deck.pptx"
PRIMARY = OUT_DIR / "primary.jpg"
GROUP = OUT_DIR / "group.jpg"
REPO_URL = "https://github.com/tattsy-maker/lang_reachy_mini"
FAIRE_URL = "https://makerfaire.com/maker/entry/reachy-mini-the-family-language-tutor-a-robot-that-knows-who-78898/"

ADVANCE_MS = 20_000  # auto-advance per slide when looping

# Palette: the explainer page's tokens.
BG = RGBColor(0xF2, 0xF4, 0xF6)
SURFACE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1B, 0x24, 0x30)
MUTED = RGBColor(0x5B, 0x66, 0x74)
LINE = RGBColor(0xCB, 0xD2, 0xDA)
LOCAL = RGBColor(0x0E, 0x7C, 0x7B)
LOCAL_SOFT = RGBColor(0xD9, 0xEE, 0xEC)
CLOUD = RGBColor(0xB9, 0x67, 0x1A)
CLOUD_SOFT = RGBColor(0xF6, 0xE5, 0xD2)
DISK = RGBColor(0x5C, 0x6B, 0x7A)
DISK_SOFT = RGBColor(0xE4, 0xE8, 0xEC)

FONT = "Arial"
MONO = "Consolas"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


# --------------------------------------------------------------------- helpers
def add_text(slide, x, y, w, h, text, size=24, bold=False, color=INK, align=PP_ALIGN.LEFT,
             font=FONT, anchor=MSO_ANCHOR.TOP, italic=False, line_spacing=1.1):
    """Add a text box. `text` may be a string or a list of paragraphs; a paragraph may be
    a string or a list of (text, {overrides}) runs."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = text if isinstance(text, list) else [text]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        runs = para if isinstance(para, list) else [(para, {})]
        for rtext, ov in runs:
            r = p.add_run()
            r.text = rtext
            f = r.font
            f.name = ov.get("font", font)
            f.size = Pt(ov.get("size", size))
            f.bold = ov.get("bold", bold)
            f.italic = ov.get("italic", italic)
            f.color.rgb = ov.get("color", color)
    return tb


def add_box(slide, x, y, w, h, fill=SURFACE, line=INK, line_w=1.25, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            radius=0.08):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    # empty text frame so the shape's default text style does not matter
    s.text_frame.text = ""
    return s


def box_text(shape, lines, size=20, color=INK, sub_color=MUTED, sub_size=None, bold_first=True,
             font=FONT):
    """Write a title line plus optional sub-lines centred in a shape."""
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    sub_size = sub_size or max(12, int(size * 0.7))
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(size if i == 0 else sub_size)
        r.font.bold = bold_first and i == 0
        r.font.color.rgb = color if i == 0 else sub_color


def add_arrow(slide, x1, y1, x2, y2, color=INK, width=1.75, dashed=False, both=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dashed:
        prst = etree.SubElement(ln, qn("a:prstDash"))
        prst.set("val", "dash")
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    if both:
        head = etree.SubElement(ln, qn("a:headEnd"))
        head.set("type", "triangle")
    return c


def add_line(slide, x1, y1, x2, y2, color=LINE, width=1.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    return c


def add_elbow(slide, points, color=INK, width=1.75, dashed=False):
    """A polyline of straight segments; arrowhead on the last one."""
    for i in range(len(points) - 1):
        (x1, y1), (x2, y2) = points[i], points[i + 1]
        last = i == len(points) - 2
        if last:
            add_arrow(slide, x1, y1, x2, y2, color=color, width=width, dashed=dashed)
        else:
            c = add_line(slide, x1, y1, x2, y2, color=color, width=width)
            if dashed:
                ln = c.line._get_or_add_ln()
                prst = etree.SubElement(ln, qn("a:prstDash"))
                prst.set("val", "dash")


def add_photo(slide, path, x, y, w, h):
    """The picture at path, or a placeholder panel when it is not on disk
    (then the caller's crop settings land on a throwaway object)."""
    if path.exists():
        return slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    add_box(slide, x, y, w, h, line=LINE, shape=MSO_SHAPE.RECTANGLE)
    return SimpleNamespace()


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def set_auto_advance(slide, ms):
    """<p:transition advTm="ms"/> between clrMapOvr and timing, per the schema order."""
    sld = slide._element
    trans = etree.SubElement(sld, qn("p:transition"))
    trans.set("spd", "med")
    trans.set("advTm", str(ms))
    # move it right after clrMapOvr (or cSld if no clrMapOvr)
    anchor = sld.find(qn("p:clrMapOvr"))
    if anchor is None:
        anchor = sld.find(qn("p:cSld"))
    anchor.addnext(trans)


def set_loop(prs):
    """Mark the show as looping and using timings (ppt/presProps.xml)."""
    for part in prs.part.package.iter_parts():
        if str(part.partname) == "/ppt/presProps.xml":
            root = etree.fromstring(part.blob)
            for old in root.findall(qn("p:showPr")):
                root.remove(old)
            show = etree.Element(qn("p:showPr"))
            show.set("loop", "1")
            show.set("useTimings", "1")
            show.set("showNarration", "1")
            etree.SubElement(show, qn("p:present"))
            etree.SubElement(show, qn("p:sldAll"))
            # showPr goes before clrMru/extLst; put it first
            root.insert(0, show)
            part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            return
    raise RuntimeError("presProps.xml not found in template")


def chrome(prs, slide, title, n, total, eyebrow="Reachy, the language tutor  ·  Maker Faire Bay Area 2026"):
    """Background, eyebrow, title, rule, slide number."""
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = BG
    add_text(slide, MARGIN, Inches(0.32), Inches(9), Inches(0.35), eyebrow, size=12, color=MUTED, font=MONO)
    add_text(slide, W - MARGIN - Inches(1.2), Inches(0.32), Inches(1.2), Inches(0.35), f"{n} / {total}",
             size=12, color=MUTED, font=MONO, align=PP_ALIGN.RIGHT)
    add_text(slide, MARGIN, Inches(0.68), W - 2 * MARGIN, Inches(0.9), title, size=38, bold=True, color=INK)
    add_line(slide, MARGIN, Inches(1.58), W - MARGIN, Inches(1.58), color=LINE, width=1.25)


def bullets(slide, x, y, w, h, items, size=22, gap=1.25, num=False, color=INK):
    paras = []
    for i, it in enumerate(items):
        lead = f"{i + 1}.  " if num else "•  "
        if isinstance(it, tuple):
            head, rest = it
            paras.append([(lead + head, {"bold": True, "color": color}), ("  " + rest, {"color": color})])
        else:
            paras.append([(lead + it, {"color": color})])
    return add_text(slide, x, y, w, h, paras, size=size, line_spacing=gap)


def chip(slide, x, y, text, fill, line, w=None, size=16):
    w = w or Inches(0.26 + 0.115 * len(text))
    s = add_box(slide, x, y, w, Inches(0.42), fill=fill, line=line, line_w=1.0, radius=0.5)
    box_text(s, [text], size=size, color=INK, bold_first=False)
    return w


# ---------------------------------------------------------------------- slides
def slide_title(prs, n, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    # photo, right half, cropped to fill
    pic_w = Inches(6.1)
    pic = add_photo(s, PRIMARY, W - pic_w, 0, pic_w, H)
    # the photo is 1047x924; fit height and crop sides to the slide half
    img_ratio = 1047 / 924
    target_ratio = pic_w / H
    if img_ratio > target_ratio:
        cut = (1 - target_ratio / img_ratio) / 2
        pic.crop_left = cut
        pic.crop_right = cut
    else:
        cut = (1 - img_ratio / target_ratio) / 2
        pic.crop_top = cut
        pic.crop_bottom = cut

    tx = MARGIN
    tw = W - pic_w - MARGIN - Inches(0.4)
    add_text(s, tx, Inches(0.9), tw, Inches(0.4), "MAKER FAIRE BAY AREA 2026  ·  ROBOTICS", size=13,
             color=MUTED, font=MONO)
    add_text(s, tx, Inches(1.4), tw, Inches(2.2), "Reachy Mini, the family language tutor", size=40,
             bold=True, color=INK, line_spacing=1.0)
    add_text(s, tx, Inches(3.7), tw, Inches(1.6),
             "A robot that knows who you are and what you're learning. It recognises your face, "
             "remembers your last lesson, and holds a spoken conversation in the language you're practising.",
             size=20, color=MUTED, line_spacing=1.2)
    add_text(s, tx, Inches(5.6), tw, Inches(0.5), "Tatiana, Yaroslav and Andrey Tsyplikhin", size=20,
             bold=True, color=INK)
    add_text(s, tx, Inches(6.05), tw, Inches(0.4), "September 25 to 27  ·  come and say hello to it", size=16,
             color=MUTED)
    add_text(s, tx, Inches(6.75), tw, Inches(0.35), "Built on a Reachy Mini by Pollen Robotics",
             size=12, color=MUTED, font=MONO)
    set_notes(s, "The loop deck starts here. If someone asks what it is: a desk robot that recognises you, "
                 "remembers what you practised last time, and tutors you out loud in your language. "
                 "Point them at slide 2 to try it.")
    return s


def slide_try_it(prs, n, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(prs, s, "Try it. It takes a minute.", n, total)
    steps = [
        ("Stand in front of the camera", "and hold still for two seconds. It greets you."),
        ("Tell it your name", "and which language you want. Say yes if it may remember you today."),
        ("Talk, then pause.", "It waits about two seconds before answering so it never cuts you off."),
        ("Say goodbye.", "Come back later and see whether it remembers you. It will."),
    ]
    bullets(s, MARGIN, Inches(1.85), Inches(8.0), Inches(4.3), steps, size=22, gap=1.3, num=True)

    # language chips
    x0 = W - MARGIN - Inches(3.9)
    add_text(s, x0, Inches(1.85), Inches(3.9), Inches(0.4), "LANGUAGES", size=12, color=MUTED, font=MONO)
    langs = ["Spanish", "French", "Italian", "Portuguese", "Russian", "Mandarin", "English"]
    x, y = x0, Inches(2.25)
    for lang in langs:
        w = Inches(0.35 + 0.115 * len(lang))
        if x + w > W - MARGIN:
            x = x0
            y += Inches(0.52)
        chip(s, x, y, lang, LOCAL_SOFT, LOCAL, w=w)
        x += w + Inches(0.12)
    add_text(s, x0, y + Inches(0.7), Inches(3.9), Inches(1.2),
             "Explanations come in your own language. Mix two languages in one sentence if you like; "
             "it does the same.", size=15, color=MUTED, line_spacing=1.2)

    # forget-me strip
    strip = add_box(s, MARGIN, Inches(6.3), W - 2 * MARGIN, Inches(0.65), fill=CLOUD_SOFT, line=CLOUD, line_w=1.0)
    box_text(strip, ['Say "forget me" at any time and it deletes your file on the spot.'], size=18,
             color=INK, bold_first=False)
    set_notes(s, "The four steps are all a visitor needs. The 2 s pause is deliberate (turn patience) so "
                 "beginners are not cut off; do not apologise for the lag, explain it. "
                 "Kids are welcome with a parent's okay.")
    return s


def slide_visit(prs, n, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(prs, s, "What happens in a visit", n, total)
    y = Inches(3.0)
    h = Inches(1.05)
    specs = [
        ("Watching", "camera on, mic muted", Inches(2.1), SURFACE, INK),
        ("Greet", "by name, or ask for it", Inches(2.3), LOCAL_SOFT, LOCAL),
        ("Tutor", "talk, correct, move", Inches(2.1), LOCAL_SOFT, LOCAL),
        ("Goodbye, save notes", "back to neutral, fresh start", Inches(3.4), SURFACE, INK),
    ]
    gap = Inches(0.75)
    total_w = sum(w for _, _, w, _, _ in specs) + gap * (len(specs) - 1)
    x = (W - total_w) / 2
    boxes = []
    for title, sub, w, fill, line in specs:
        b = add_box(s, x, y, w, h, fill=fill, line=line, line_w=1.5, radius=0.5)
        box_text(b, [title, sub], size=22, sub_size=14)
        boxes.append((x, w))
        x += w + gap
    labels = [
        ["face steady", "about 2 s"],
        ["known face:", "loads last notes"],
        ["goodbye, or gone", "a minute, or a new voice"],
    ]
    for i in range(3):
        x1 = boxes[i][0] + boxes[i][1]
        x2 = boxes[i + 1][0]
        add_arrow(s, x1, y + h / 2, x2, y + h / 2)
        add_text(s, x1 - Inches(0.6), y - Inches(0.85), (x2 - x1) + Inches(1.2), Inches(0.8), labels[i],
                 size=13, color=MUTED, align=PP_ALIGN.CENTER, line_spacing=1.0, anchor=MSO_ANCHOR.BOTTOM)
    # return loop
    lx = boxes[3][0] + boxes[3][1] / 2
    rx = boxes[0][0] + boxes[0][1] / 2
    ly = y + h + Inches(0.75)
    add_elbow(s, [(lx, y + h), (lx, ly), (rx, ly), (rx, y + h)])
    add_text(s, rx, ly + Inches(0.05), lx - rx, Inches(0.4), "the next face starts a brand-new session",
             size=14, color=MUTED, align=PP_ALIGN.CENTER)

    foot = [
        "A passer-by is not a visitor: the face has to hold still first.",
        "Speaking counts as presence, so stepping out of frame mid-sentence never ends a session.",
        "After 40 s of silence it asks \"still there?\". Nobody for two minutes: a short dance, then back to watching.",
    ]
    bullets(s, MARGIN, Inches(5.55), W - 2 * MARGIN, Inches(1.6), foot, size=16, gap=1.2, color=MUTED)
    set_notes(s, "Nothing here is started by hand. The runner watches the camera; a steady face starts a "
                 "session, goodbye / walk-away / a different face ends it and writes the notes. "
                 "If it seems to ignore someone, they are probably moving or off-centre.")
    return s


def slide_knows_you(prs, n, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(prs, s, "How it knows you", n, total)
    cards = [
        ("Known face", "Greets you by name and opens with what you struggled with last time. "
                       "Your profile and the last three session notes are read before it speaks."),
        ("Unsure match", "Asks \"Maria, is that you?\" instead of guessing. A wrong name is the worst mistake "
                         "this feature can make, so it checks the face before believing a yes."),
        ("Stranger", "Asks your name, your language and your goal, offers to remember you for the day, "
                     "and takes a face and voice signature during that chat."),
    ]
    cw = (W - 2 * MARGIN - 2 * Inches(0.35)) / 3
    x = MARGIN
    for title, body in cards:
        b = add_box(s, x, Inches(1.9), cw, Inches(2.65), fill=SURFACE, line=LINE, line_w=1.0)
        add_text(s, x + Inches(0.2), Inches(2.05), cw - Inches(0.4), Inches(0.5), title, size=24, bold=True,
                 color=LOCAL)
        add_text(s, x + Inches(0.2), Inches(2.6), cw - Inches(0.4), Inches(1.9), body, size=16, color=INK,
                 line_spacing=1.2)
        x += cw + Inches(0.35)

    strip = add_box(s, MARGIN, Inches(4.85), W - 2 * MARGIN, Inches(2.15), fill=DISK_SOFT, line=DISK, line_w=1.0)
    add_text(s, MARGIN + Inches(0.25), Inches(4.95), Inches(4), Inches(0.4), "WHAT IT KEEPS, AND WHERE",
             size=12, color=MUTED, font=MONO)
    priv = [
        ("Only if you say yes.", "Otherwise nothing is stored."),
        ("Your first name, your language, its lesson notes,", "and a numeric face signature and voice "
                                                              "signature. Not a photo, not a recording, "
                                                              "and neither can be turned back into one."),
        ("On the computer at this booth.", "Face data never leaves it. Every visitor profile is deleted "
                                           "at the end of the day."),
    ]
    bullets(s, MARGIN + Inches(0.25), Inches(5.3), W - 2 * MARGIN - Inches(0.5), Inches(1.7), priv, size=16,
            gap=1.2)
    set_notes(s, "The privacy strip is the signage's disclosure in short. If asked about photos: the face "
                 "signature is a vector of numbers computed on this box; the look tool's snapshots and the "
                 "session logs stay on this machine and guests are wiped at shutdown.")
    return s


def slide_memory(prs, n, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(prs, s, "What it remembers", n, total)
    tree = add_box(s, MARGIN, Inches(1.9), Inches(6.2), Inches(4.9), fill=SURFACE, line=LINE, line_w=1.0)
    code = [
        [("learners/", {"color": INK, "bold": True})],
        [("  maria/", {"color": INK, "bold": True})],
        [("    profile.json", {"color": LOCAL, "bold": True})],
        [("      name, target language, level, goal,", {"color": MUTED})],
        [("      face signature, voice signature,", {"color": MUTED})],
        [("      family or guest", {"color": MUTED})],
        [("", {})],
        [("    notes.md", {"color": LOCAL, "bold": True})],
        [("      written after each session:", {"color": MUTED})],
        [("      practised, struggled with, wins,", {"color": MUTED})],
        [("      next time", {"color": MUTED})],
    ]
    add_text(s, MARGIN + Inches(0.25), Inches(2.1), Inches(5.8), Inches(4.5), code, size=15, font=MONO,
             line_spacing=1.3)

    x = MARGIN + Inches(6.6)
    w = W - MARGIN - x
    items = [
        ("Plain files, one folder per person.", "Readable and editable by a human. No database, "
                                                "nothing in the cloud."),
        ("The notes are prose.", "The tutor writes them and reads them back next time. That is the form "
                                 "a language model uses best."),
        ("Family profiles stay.", "Guest profiles, everyone enrolled at the booth, are wiped when the "
                                  "booth shuts down."),
    ]
    bullets(s, x, Inches(1.95), w, Inches(4.8), items, size=18, gap=1.3)
    set_notes(s, "Open a learners/<name>/notes.md on the laptop if someone wants to see what it wrote. "
                 "The face signature in profile.json is a list of numbers, which usually settles the "
                 "privacy question.")
    return s


def slide_alive(prs, n, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(prs, s, "Why it feels alive", n, total)
    add_text(s, MARGIN, Inches(1.8), W - 2 * MARGIN, Inches(0.6),
             "Three things move the robot at once. They never fight because each owns its own joints.",
             size=20, color=MUTED)
    rows = [
        ("Face tracker", "head turn, body turn", "Where your face is in the camera frame. Calm on purpose: "
                                                 "small steps, about once a second.", LOCAL_SOFT, LOCAL),
        ("Embodiment", "head tilt, antennas", "You starting or stopping speaking, the robot starting or "
                                              "stopping speaking. Leans in, dips, sways.", LOCAL_SOFT, LOCAL),
        ("The brain's tools", "anything, briefly", "An explicit gesture or dance. The tracker pauses while "
                                                   "it runs and re-reads the pose afterwards.", CLOUD_SOFT,
         CLOUD),
    ]
    hx = [MARGIN, MARGIN + Inches(2.9), MARGIN + Inches(5.7)]
    hw = [Inches(2.7), Inches(2.6), W - MARGIN - hx[2]]
    for x, w, t in zip(hx, hw, ["MOVER", "OWNS", "TRIGGERED BY"]):
        add_text(s, x, Inches(2.55), w, Inches(0.35), t, size=12, color=MUTED, font=MONO)
    y = Inches(2.95)
    rh = Inches(0.95)
    for mover, owns, trig, fill, line in rows:
        b = add_box(s, hx[0], y, hw[0], rh - Inches(0.15), fill=fill, line=line, line_w=1.25)
        box_text(b, [mover], size=20)
        add_text(s, hx[1], y, hw[1], rh, owns, size=18, bold=True, color=INK, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, hx[2], y, hw[2], rh, trig, size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
        y += rh
    punch = add_box(s, MARGIN, Inches(6.0), W - 2 * MARGIN, Inches(0.95), fill=SURFACE, line=LINE, line_w=1.0)
    box_text(punch, ["It reacts before it answers.", "Moving on every event costs no model round trip, "
                                                     "which is what makes it read as listening."],
             size=22, sub_size=16)
    set_notes(s, "The tracker follows your face slowly; the body language reacts to who is talking; "
                 "dances and nods are the model's own tool calls. If the base twitches by itself after a "
                 "dance, that is the servo settling, not the software.")
    return s


def slide_under_hood(prs, n, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(prs, s, "Under the hood", n, total)
    # process diagram
    y = Inches(2.5)
    h = Inches(1.05)
    procs = [
        ("Voice agent", "hears, talks, decides", LOCAL_SOFT, LOCAL),
        ("Robot driver", "safe named moves, e-stop", LOCAL_SOFT, LOCAL),
        ("Vendor daemon", "owns USB, mic, camera", LOCAL_SOFT, LOCAL),
        ("Reachy Mini", "9 motors, camera, mic, speaker", SURFACE, INK),
    ]
    bw = Inches(2.55)
    gap = Inches(0.6)
    x = MARGIN
    xs = []
    for title, sub, fill, line in procs:
        b = add_box(s, x, y, bw, h, fill=fill, line=line, line_w=1.5)
        box_text(b, [title, sub], size=19, sub_size=13)
        xs.append(x)
        x += bw + gap
    for i, lab in enumerate(["Device Connect", "vendor API", "USB serial"]):
        x1 = xs[i] + bw
        x2 = xs[i + 1]
        add_arrow(s, x1, y + h / 2, x2, y + h / 2)
        add_text(s, x1 - Inches(0.7), y + h + Inches(0.02), gap + Inches(1.4), Inches(0.3), lab, size=11,
                 color=MUTED, align=PP_ALIGN.CENTER, font=MONO)
    # brain above the agent
    brain = add_box(s, xs[0], Inches(1.72), bw, Inches(0.62), fill=CLOUD_SOFT, line=CLOUD, line_w=1.5)
    box_text(brain, ["the brain, in the cloud", "the only thing that leaves this machine"], size=14, sub_size=10)
    add_arrow(s, xs[0] + bw / 2, Inches(2.34), xs[0] + bw / 2, y, both=True, width=1.5)
    # learners below the agent
    disk = add_box(s, xs[0], y + h + Inches(0.55), bw, Inches(0.5), fill=DISK_SOFT, line=DISK, line_w=1.25)
    box_text(disk, ["learners/ folder on disk"], size=15, bold_first=False)
    add_arrow(s, xs[0] + bw / 2, y + h, xs[0] + bw / 2, y + h + Inches(0.55), both=True, width=1.5)
    # media loan
    add_elbow(s, [(xs[2] + bw / 2, y + h), (xs[2] + bw / 2, y + h + Inches(0.4)),
                  (xs[1] + bw / 2, y + h + Inches(0.4)), (xs[0] + bw + Inches(0.05), y + h + Inches(0.4))],
              color=MUTED, dashed=True, width=1.25)
    add_text(s, xs[1] - Inches(0.2), y + h + Inches(0.45), bw + Inches(1.6), Inches(0.35),
             "lends the mic, speaker and camera to the agent", size=12, color=MUTED, align=PP_ALIGN.CENTER)
    # one turn, two modes
    ty = Inches(4.85)
    add_text(s, MARGIN, ty, Inches(6), Inches(0.35), "ONE TURN OF CONVERSATION", size=12, color=MUTED, font=MONO)
    rows = [
        ("Cloud voice  ·  booth default", "under 1 s",
         "Audio streams to Google's Gemini Live, which hears, thinks and speaks. Mixes languages natively; "
         "you can interrupt it.", CLOUD_SOFT, CLOUD),
        ("Local voice  ·  fallback if the hall's internet dies", "3 to 5 s",
         "Whisper on the GPU, then Claude gets only the text, then Kokoro or Piper speaks. "
         "No audio leaves the booth.", LOCAL_SOFT, LOCAL),
    ]
    ry = ty + Inches(0.4)
    for head, lag, body, fill, line in rows:
        b = add_box(s, MARGIN, ry, Inches(4.3), Inches(0.8), fill=fill, line=line, line_w=1.25)
        box_text(b, [head, lag], size=15, sub_size=14)
        add_text(s, MARGIN + Inches(4.5), ry, W - 2 * MARGIN - Inches(4.5), Inches(0.8), body, size=15,
                 color=INK, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
        ry += Inches(0.9)
    set_notes(s, "Three processes on the booth computer: our voice agent, our robot driver (the only thing "
                 "allowed to own the robot), and Pollen's daemon on the USB bus. Device Connect is the "
                 "messaging layer between agent and driver, so the robot could sit on another computer. "
                 "Cloud mode is what is running today; face data never leaves the booth either way.")
    return s


def slide_built_with(prs, n, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(prs, s, "Built with, and a question for you", n, total)
    # group photo, left, cropped square
    pw = Inches(3.4)
    pic = add_photo(s, GROUP, MARGIN, Inches(1.9), pw, Inches(4.9))
    img_ratio = 1920 / 2560
    target_ratio = pw / Inches(4.9)
    cut = (1 - target_ratio / img_ratio) / 2 if img_ratio > target_ratio else 0
    pic.crop_left = pic.crop_right = cut
    if img_ratio <= target_ratio:
        cut = (1 - img_ratio / target_ratio) / 2
        pic.crop_top = pic.crop_bottom = cut
    add_text(s, MARGIN, Inches(6.85), pw, Inches(0.4), "Tatiana, Yaroslav and Andrey", size=13, color=MUTED,
             align=PP_ALIGN.CENTER)

    x = MARGIN + pw + Inches(0.45)
    w = W - MARGIN - Inches(2.6) - x
    stack = [
        ("Robot", "Reachy Mini by Pollen Robotics"),
        ("Computer", "NVIDIA DGX Spark (GB10), everything but the brain runs here"),
        ("Voices", "Gemini Live in the cloud; Whisper, Kokoro and Piper locally"),
        ("Brains", "Gemini Live, or Claude in the local fallback"),
        ("Recognition", "face embeddings, SpeechBrain voice prints"),
        ("Plumbing", "pipecat for the conversation, Device Connect between agent and robot"),
    ]
    bullets(s, x, Inches(1.9), w, Inches(3.3), stack, size=16, gap=1.25)
    ask = add_box(s, x, Inches(5.35), w, Inches(1.5), fill=CLOUD_SOFT, line=CLOUD, line_w=1.25)
    box_text(ask, ["What would you want it to do?", "Tell the robot, or tell us. It writes wishes down."],
             size=22, sub_size=15)

    # QR to the repo
    qx = W - MARGIN - Inches(2.3)
    qr_png = OUT_DIR / "qr_repo.png"
    img = qrcode.make(REPO_URL, box_size=10, border=2)
    img.save(qr_png)
    s.shapes.add_picture(str(qr_png), qx, Inches(1.95), width=Inches(2.3), height=Inches(2.3))
    add_text(s, qx - Inches(0.3), Inches(4.3), Inches(2.9), Inches(0.9),
             ["the code, open source", "github.com/tattsy-maker/lang_reachy_mini"], size=12, color=MUTED,
             align=PP_ALIGN.CENTER, font=MONO, line_spacing=1.2)
    set_notes(s, "Everything is open on GitHub. Ask visitors the wishlist question; the robot also asks it "
                 "after goodbye and the answers land in booth/feedback.md. "
                 "Maker Faire entry: " + FAIRE_URL)
    return s


# ------------------------------------------------------------------------ main
def main():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    builders = [slide_title, slide_try_it, slide_visit, slide_knows_you, slide_memory, slide_alive,
                slide_under_hood, slide_built_with]
    total = len(builders)
    for i, build in enumerate(builders, start=1):
        slide = build(prs, i, total)
        set_auto_advance(slide, ADVANCE_MS)
    set_loop(prs)
    OUT_DIR.mkdir(exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT} ({total} slides, auto-advance {ADVANCE_MS // 1000}s, looping)")


if __name__ == "__main__":
    main()
