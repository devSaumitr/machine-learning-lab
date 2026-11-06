"""
PDF Framework for ML Notes — professional textbook-quality generation.
Follows AGENT_STYLE teaching methodology.
"""
import os
import io
import re
import textwrap
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle, KeepTogether, HRFlowable, ListFlowable, ListItem,
    NextPageTemplate, PageTemplate, BaseDocTemplate, Frame
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle, Polygon, Group
from reportlab.graphics import renderPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ─── Colors ──────────────────────────────────────────────────────────────────
PRIMARY = HexColor('#1a5276')      # dark blue
SECONDARY = HexColor('#2e86c1')    # medium blue
ACCENT = HexColor('#e74c3c')       # red for warnings
CODE_BG = HexColor('#f4f6f7')
CODE_BORDER = HexColor('#d5dbdb')
TABLE_HEADER_BG = HexColor('#2c3e50')
TABLE_ALT_BG = HexColor('#eaf2f8')
SUCCESS = HexColor('#27ae60')
WARNING_BG = HexColor('#fef9e7')
WARNING_BORDER = HexColor('#f39c12')
TIP_BG = HexColor('#eafaf1')
TIP_BORDER = HexColor('#27ae60')
NOTE_BG = HexColor('#eaf2f8')
NOTE_BORDER = HexColor('#2980b9')
LIGHT_GRAY = HexColor('#f8f9f9')
MEDIUM_GRAY = HexColor('#bdc3c7')
DARK = HexColor('#2c3e50')

# ─── Register fonts ──────────────────────────────────────────────────────────
def register_fonts():
    """Try to register good fonts, fall back to Helvetica."""
    font_dir_candidates = [
        r'C:\Windows\Fonts',
        '/usr/share/fonts/truetype/dejavu',
        '/usr/share/fonts/truetype/liberation',
    ]
    registered = False
    for fd in font_dir_candidates:
        if not os.path.isdir(fd):
            continue
        # Try DejaVu
        for name, fname in [
            ('DejaVu', 'DejaVuSans.ttf'),
            ('DejaVu-Bold', 'DejaVuSans-Bold.ttf'),
            ('DejaVu-Oblique', 'DejaVuSans-Oblique.ttf'),
            ('DejaVuMono', 'DejaVuSansMono.ttf'),
        ]:
            path = os.path.join(fd, fname)
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont(name, path))
                    registered = True
                except Exception:
                    pass
        if registered:
            break
        # Try Consolas / Calibri on Windows
        for name, fname in [
            ('Body', 'calibri.ttf'),
            ('Body-Bold', 'calibrib.ttf'),
            ('Mono', 'consola.ttf'),
        ]:
            path = os.path.join(fd, fname)
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont(name, path))
                    registered = True
                except Exception:
                    pass
        if registered:
            break
    return registered

HAS_CUSTOM = register_fonts()
BODY_FONT = 'DejaVu' if HAS_CUSTOM and 'DejaVu' in [f.fontName for f in pdfmetrics.getRegisteredFontNames() and []] or True else 'Helvetica'
# Simpler: just use Helvetica family which is built-in and reliable
BODY_FONT = 'Helvetica'
BODY_BOLD = 'Helvetica-Bold'
BODY_ITALIC = 'Helvetica-Oblique'
MONO_FONT = 'Courier'
MONO_BOLD = 'Courier-Bold'

# ─── Math rendering ──────────────────────────────────────────────────────────
_math_cache = {}

def render_math(tex, fontsize=11, color='#000000', dpi=200):
    """Render LaTeX math to a PNG image path using matplotlib."""
    key = (tex, fontsize, color)
    if key in _math_cache:
        return _math_cache[key]
    
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.patch.set_alpha(0)
    try:
        t = fig.text(0, 0, f'${tex}$', fontsize=fontsize, color=color,
                     math_fontset='cm', ha='left', va='bottom')
        fig.canvas.draw()
        bbox = t.get_window_extent()
        w, h = bbox.width / dpi, bbox.height / dpi
        fig.set_size_inches(w + 0.1, h + 0.1)
        t.set_position((0.05 / (w + 0.1), 0.05 / (h + 0.1)))
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, transparent=True,
                    bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        buf.seek(0)
        _math_cache[key] = buf
        return buf
    except Exception as e:
        plt.close(fig)
        # Fallback: plain text
        buf = io.BytesIO()
        fig2 = plt.figure(figsize=(0.1, 0.1))
        fig2.patch.set_alpha(0)
        t = fig2.text(0, 0, tex, fontsize=fontsize, color=color, ha='left', va='bottom')
        fig2.canvas.draw()
        bbox = t.get_window_extent()
        w, h = bbox.width / dpi, bbox.height / dpi
        fig2.set_size_inches(w + 0.1, h + 0.1)
        t.set_position((0.05 / (w + 0.1), 0.05 / (h + 0.1)))
        fig2.savefig(buf, format='png', dpi=dpi, transparent=True,
                     bbox_inches='tight', pad_inches=0.02)
        plt.close(fig2)
        buf.seek(0)
        _math_cache[key] = buf
        return buf

def math_flowable(tex, fontsize=11, centered=True, color='#000000'):
    """Create a reportlab flowable for a math expression."""
    buf = render_math(tex, fontsize, color)
    # Get image size
    from PIL import Image as PILImage
    buf.seek(0)
    img = PILImage.open(buf)
    pw, ph = img.size
    # Scale to points (1 inch = 72 points), keep readable size
    scale = min(1.0, 400 / pw)  # max width ~400pt
    disp_w = pw * 0.45 * scale   # 0.45 ≈ 72/160 for reasonable size
    disp_h = ph * 0.45 * scale
    buf.seek(0)
    img_f = Image(buf, width=disp_w, height=disp_h)
    if centered:
        t = Table([[img_f]], colWidths=[None])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return t
    return img_f

def display_math(tex, label=None):
    """Display equation with optional right-aligned label like (31.1)."""
    math_img = math_flowable(tex, fontsize=12)
    if label:
        lbl = Paragraph(f'<font size=9 color="#555555">({label})</font>',
                       ParagraphStyle('MathLabel', fontName=BODY_FONT, fontSize=9,
                                      alignment=TA_RIGHT))
        t = Table([[math_img, lbl]], colWidths=[None, 60])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        return t
    return math_img

def inline_math_html(tex, fontsize=10):
    """Render inline math as an <img> tag for use inside Paragraph."""
    buf = render_math(tex, fontsize)
    from PIL import Image as PILImage
    buf.seek(0)
    img = PILImage.open(buf)
    pw, ph = img.size
    scale = 12.0 / ph  # target height ~12pt for inline
    disp_w = pw * scale
    disp_h = 12.0
    # Save to temp file for reportlab img tag
    import hashlib
    h = hashlib.md5(tex.encode()).hexdigest()[:12]
    path = os.path.join(FIGDIR, f'_inline_{h}.png')
    buf.seek(0)
    with open(path, 'wb') as f:
        f.write(buf.read())
    return f'<img src="{path}" width="{disp_w:.1f}" height="{disp_h:.1f}" valign="middle"/>'

# ─── Directories ─────────────────────────────────────────────────────────────
BASE_DIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40'
FIGDIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ─── Styles ──────────────────────────────────────────────────────────────────
def get_styles():
    s = {}
    s['title'] = ParagraphStyle(
        'DocTitle', fontName=BODY_BOLD, fontSize=26, leading=32,
        textColor=PRIMARY, alignment=TA_CENTER, spaceAfter=6
    )
    s['subtitle'] = ParagraphStyle(
        'DocSubtitle', fontName=BODY_FONT, fontSize=13, leading=18,
        textColor=SECONDARY, alignment=TA_CENTER, spaceAfter=20
    )
    s['chapter_num'] = ParagraphStyle(
        'ChapterNum', fontName=BODY_BOLD, fontSize=14, leading=18,
        textColor=SECONDARY, alignment=TA_CENTER, spaceAfter=4
    )
    s['h1'] = ParagraphStyle(
        'H1', fontName=BODY_BOLD, fontSize=18, leading=24,
        textColor=PRIMARY, spaceBefore=18, spaceAfter=10,
        borderPadding=(0, 0, 4, 0)
    )
    s['h2'] = ParagraphStyle(
        'H2', fontName=BODY_BOLD, fontSize=14, leading=19,
        textColor=DARK, spaceBefore=14, spaceAfter=7
    )
    s['h3'] = ParagraphStyle(
        'H3', fontName=BODY_BOLD, fontSize=12, leading=16,
        textColor=HexColor('#34495e'), spaceBefore=10, spaceAfter=5
    )
    s['h4'] = ParagraphStyle(
        'H4', fontName=BODY_BOLD, fontSize=10.5, leading=14,
        textColor=HexColor('#5d6d7e'), spaceBefore=8, spaceAfter=4
    )
    s['body'] = ParagraphStyle(
        'Body', fontName=BODY_FONT, fontSize=10, leading=14.5,
        textColor=black, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=5
    )
    s['body_center'] = ParagraphStyle(
        'BodyCenter', parent=s['body'], alignment=TA_CENTER
    )
    s['bullet'] = ParagraphStyle(
        'Bullet', fontName=BODY_FONT, fontSize=10, leading=14,
        leftIndent=18, bulletIndent=6, spaceBefore=2, spaceAfter=2,
        textColor=black
    )
    s['bullet2'] = ParagraphStyle(
        'Bullet2', fontName=BODY_FONT, fontSize=9.5, leading=13,
        leftIndent=36, bulletIndent=22, spaceBefore=1, spaceAfter=1,
        textColor=HexColor('#333333')
    )
    s['code'] = ParagraphStyle(
        'Code', fontName=MONO_FONT, fontSize=8.5, leading=11.5,
        textColor=HexColor('#1a1a1a'), backColor=CODE_BG,
        borderPadding=(8, 8, 8, 8), spaceBefore=4, spaceAfter=4,
        leftIndent=6, rightIndent=6
    )
    s['code_caption'] = ParagraphStyle(
        'CodeCaption', fontName=BODY_ITALIC, fontSize=8, leading=11,
        textColor=HexColor('#666666'), alignment=TA_LEFT,
        spaceBefore=1, spaceAfter=6, leftIndent=6
    )
    s['caption'] = ParagraphStyle(
        'Caption', fontName=BODY_ITALIC, fontSize=8.5, leading=12,
        textColor=HexColor('#555555'), alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=10
    )
    s['table_header'] = ParagraphStyle(
        'TableHeader', fontName=BODY_BOLD, fontSize=9, leading=12,
        textColor=white, alignment=TA_CENTER
    )
    s['table_cell'] = ParagraphStyle(
        'TableCell', fontName=BODY_FONT, fontSize=9, leading=12,
        textColor=black, alignment=TA_LEFT
    )
    s['table_cell_center'] = ParagraphStyle(
        'TableCellC', fontName=BODY_FONT, fontSize=9, leading=12,
        textColor=black, alignment=TA_CENTER
    )
    s['note'] = ParagraphStyle(
        'Note', fontName=BODY_FONT, fontSize=9.5, leading=13.5,
        textColor=HexColor('#1a5276'), leftIndent=10, rightIndent=10,
        spaceBefore=4, spaceAfter=4
    )
    s['warning'] = ParagraphStyle(
        'Warning', fontName=BODY_FONT, fontSize=9.5, leading=13.5,
        textColor=HexColor('#935116'), leftIndent=10, rightIndent=10,
        spaceBefore=4, spaceAfter=4
    )
    s['tip'] = ParagraphStyle(
        'Tip', fontName=BODY_FONT, fontSize=9.5, leading=13.5,
        textColor=HexColor('#1e8449'), leftIndent=10, rightIndent=10,
        spaceBefore=4, spaceAfter=4
    )
    s['takeaway'] = ParagraphStyle(
        'Takeaway', fontName=BODY_BOLD, fontSize=10.5, leading=15,
        textColor=PRIMARY, alignment=TA_LEFT,
        spaceBefore=10, spaceAfter=10, leftIndent=8, rightIndent=8
    )
    s['toc_h1'] = ParagraphStyle(
        'TOCH1', fontName=BODY_BOLD, fontSize=11, leading=16,
        textColor=PRIMARY, spaceBefore=6, spaceAfter=2
    )
    s['toc_h2'] = ParagraphStyle(
        'TOCH2', fontName=BODY_FONT, fontSize=10, leading=14,
        textColor=DARK, leftIndent=18, spaceBefore=1, spaceAfter=1
    )
    s['footer'] = ParagraphStyle(
        'Footer', fontName=BODY_FONT, fontSize=8, leading=10,
        textColor=HexColor('#888888'), alignment=TA_CENTER
    )
    return s

STYLES = get_styles()

# ─── Content builders ────────────────────────────────────────────────────────
def H1(text, bookmark=None):
    """Section heading level 1 with colored bar."""
    bar = HRFlowable(width="100%", thickness=2, color=PRIMARY,
                     spaceBefore=2, spaceAfter=6)
    p = Paragraph(text, STYLES['h1'])
    return [bar, p]

def H2(text):
    return Paragraph(text, STYLES['h2'])

def H3(text):
    return Paragraph(text, STYLES['h3'])

def H4(text):
    return Paragraph(text, STYLES['h4'])

def P(text, style=None):
    """Body paragraph. Supports basic HTML markup."""
    return Paragraph(text, style or STYLES['body'])

def B(text):
    """Bold run helper."""
    return f'<b>{text}</b>'

def I(text):
    return f'<i>{text}</i>'

def bullets(items, style=None):
    """Create bullet list flowables."""
    st = style or STYLES['bullet']
    result = []
    for item in items:
        result.append(Paragraph(f'• {item}', st))
    return result

def numbered(items, style=None):
    st = style or STYLES['bullet']
    result = []
    for i, item in enumerate(items, 1):
        result.append(Paragraph(f'{i}. {item}', st))
    return result

def code_block(code, caption=None):
    """Formatted code block with background — allows page breaks."""
    # Escape XML
    escaped = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Preserve newlines as <br/>
    escaped = escaped.replace('\n', '<br/>')
    
    elements = []
    # Use Paragraph with backColor (allows splitting across pages)
    code_p = Paragraph(escaped, STYLES['code'])
    elements.append(code_p)
    if caption:
        elements.append(Paragraph(caption, STYLES['code_caption']))
    return elements

def simple_highlight(escaped_code):
    """No-op highlighting — code is already escaped. Just return as-is."""
    return escaped_code

def note_box(title, text):
    """Blue informational note box."""
    content = Paragraph(f'<b>{title}</b><br/>{text}', STYLES['note'])
    t = Table([[content]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NOTE_BG),
        ('BOX', (0, 0), (-1, -1), 1, NOTE_BORDER),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t

def warning_box(title, text):
    content = Paragraph(f'<b>{title}</b><br/>{text}', STYLES['warning'])
    t = Table([[content]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), WARNING_BG),
        ('BOX', (0, 0), (-1, -1), 1, WARNING_BORDER),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t

def tip_box(title, text):
    content = Paragraph(f'<b>{title}</b><br/>{text}', STYLES['tip'])
    t = Table([[content]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), TIP_BG),
        ('BOX', (0, 0), (-1, -1), 1, TIP_BORDER),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t

def takeaway(text):
    """Bold lock-in takeaway line."""
    content = Paragraph(f'&#9733; {text}', STYLES['takeaway'])
    t = Table([[content]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#eaf2f8')),
        ('LINEBEFORE', (0, 0), (0, -1), 3, PRIMARY),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t

def make_table(headers, rows, col_widths=None, caption=None):
    """Professional data table."""
    header_cells = [Paragraph(h, STYLES['table_header']) for h in headers]
    data = [header_cells]
    for row in rows:
        data.append([Paragraph(str(c), STYLES['table_cell']) for c in row])
    
    if col_widths is None:
        n = len(headers)
        col_widths = [None] * n
    
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), BODY_BOLD),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, MEDIUM_GRAY),
        ('BOX', (0, 0), (-1, -1), 1, DARK),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
    ]
    # Alternate row colors
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), TABLE_ALT_BG))
    t.setStyle(TableStyle(style_cmds))
    
    result = [t]
    if caption:
        result.append(Paragraph(caption, STYLES['caption']))
    return result

def figure(path, width=440, caption=None):
    """Insert a matplotlib-generated figure."""
    from PIL import Image as PILImage
    img = PILImage.open(path)
    pw, ph = img.size
    aspect = ph / pw
    disp_w = min(width, 460)
    disp_h = disp_w * aspect
    img_f = Image(path, width=disp_w, height=disp_h)
    result = [Spacer(1, 4), img_f]
    if caption:
        result.append(Paragraph(caption, STYLES['caption']))
    return result

def math_block(tex, label=None):
    """Centered display math."""
    return display_math(tex, label)

# ─── Page templates with header/footer ───────────────────────────────────────
class MLDocTemplate(BaseDocTemplate):
    """Custom document template with headers, footers, TOC support."""
    def __init__(self, filename, **kwargs):
        self.chapter_title = kwargs.pop('chapter_title', '')
        self.book_title = kwargs.pop('book_title',
            'Machine Learning Notes — CampusX 100 Days of ML')
        BaseDocTemplate.__init__(self, filename, **kwargs)
        
        frame_normal = Frame(
            self.leftMargin, self.bottomMargin,
            self.width, self.height,
            id='normal',
            leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0
        )
        frame_first = Frame(
            self.leftMargin, self.bottomMargin,
            self.width, self.height,
            id='first',
            leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0
        )
        
        self.addPageTemplates([
            PageTemplate(id='First', frames=[frame_first],
                        onPage=self._first_page),
            PageTemplate(id='Normal', frames=[frame_normal],
                        onPage=self._later_page),
        ])
    
    def _draw_header_footer(self, canvas, doc, is_first=False):
        canvas.saveState()
        w, h = A4
        
        if not is_first:
            # Header line
            canvas.setStrokeColor(PRIMARY)
            canvas.setLineWidth(0.8)
            canvas.line(50, h - 40, w - 50, h - 40)
            canvas.setFont(BODY_FONT, 7.5)
            canvas.setFillColor(HexColor('#666666'))
            header_text = self.chapter_title if self.chapter_title else self.book_title
            canvas.drawString(50, h - 36, header_text[:80])
            canvas.drawRightString(w - 50, h - 36, self.book_title[:50])
        
        # Footer
        canvas.setStrokeColor(MEDIUM_GRAY)
        canvas.setLineWidth(0.5)
        canvas.line(50, 38, w - 50, 38)
        canvas.setFont(BODY_FONT, 8)
        canvas.setFillColor(HexColor('#888888'))
        canvas.drawCentredString(w / 2, 26, f'Page {doc.page}')
        canvas.drawString(50, 26, 'CampusX — 100 Days of Machine Learning')
        canvas.drawRightString(w - 50, 26, 'Videos 31–40')
        
        canvas.restoreState()
    
    def _first_page(self, canvas, doc):
        self._draw_header_footer(canvas, doc, is_first=True)
    
    def _later_page(self, canvas, doc):
        self._draw_header_footer(canvas, doc, is_first=False)
    
    def afterFlowable(self, flowable):
        """Notify TOC of headings."""
        if hasattr(flowable, 'style'):
            style_name = flowable.style.name if hasattr(flowable.style, 'name') else ''
            text = flowable.getPlainText() if hasattr(flowable, 'getPlainText') else ''
            if style_name == 'H1':
                self.notify('TOCEntry', (0, text, self.page))
            elif style_name == 'H2':
                self.notify('TOCEntry', (1, text, self.page))

def build_toc():
    """Create a table of contents flowable."""
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle('TOC0', fontName=BODY_BOLD, fontSize=10.5, leading=18,
                       textColor=PRIMARY, spaceBefore=6),
        ParagraphStyle('TOC1', fontName=BODY_FONT, fontSize=9.5, leading=14,
                       textColor=DARK, leftIndent=20, spaceBefore=1),
    ]
    return toc

def page_break():
    return PageBreak()

def spacer(h=10):
    return Spacer(1, h)

def section_divider():
    return HRFlowable(width="100%", thickness=0.5, color=MEDIUM_GRAY,
                      spaceBefore=8, spaceAfter=8)
