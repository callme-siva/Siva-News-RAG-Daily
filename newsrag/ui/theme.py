"""Appearance: font size (A / A+ / A++) and theme (light / dark / system), applied with CSS.

Streamlit sizes its UI in rem, so setting the root font size scales the whole app.
Colours are CSS variables; "system" follows the operating system via prefers-color-scheme.
"""

from __future__ import annotations

from newsrag.settings import FONT_SIZES_PX

LIGHT = {
    "bg": "#ffffff",
    "bg2": "#f5f5f2",
    "text": "#1f1f1d",
    "muted": "#5f5e5a",
    "border": "rgba(31,31,29,0.15)",
    "accent": "#185fa5",
    "accent_bg": "#e6f1fb",
    "ok": "#27500a",
    "ok_bg": "#eaf3de",
}
DARK = {
    "bg": "#1f1f1d",
    "bg2": "#2c2c2a",
    "text": "#f1efe8",
    "muted": "#b4b2a9",
    "border": "rgba(241,239,232,0.18)",
    "accent": "#85b7eb",
    "accent_bg": "#0c447c",
    "ok": "#c0dd97",
    "ok_bg": "#27500a",
}


def _vars(palette: dict[str, str], high_contrast: bool) -> str:
    p = dict(palette)
    if high_contrast:
        p["muted"] = p["text"]
        p["border"] = p["text"]
    return "".join(f"--nr-{k.replace('_', '-')}:{v};" for k, v in p.items())


def css(font_size: str, theme: str, high_contrast: bool = False) -> str:
    px = FONT_SIZES_PX.get(font_size, 16)
    light, dark = _vars(LIGHT, high_contrast), _vars(DARK, high_contrast)
    if theme == "light":
        palette = f":root{{{light}}}"
    elif theme == "dark":
        palette = f":root{{{dark}}}"
    else:
        palette = f":root{{{light}}}@media (prefers-color-scheme: dark){{:root{{{dark}}}}}"
    return f"""<style>
{palette}
html {{ font-size: {px}px; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
  background: var(--nr-bg); color: var(--nr-text);
}}
[data-testid="stSidebar"] {{ background: var(--nr-bg2); }}
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp td, .stApp th, [data-testid="stMarkdownContainer"] {{ color: var(--nr-text); }}
.stApp a {{ color: var(--nr-accent); }}
.nr-muted {{ color: var(--nr-muted) !important; font-size: 0.9rem; }}
.nr-badge {{ display:inline-block; padding:2px 10px; border-radius:8px; font-size:0.8rem;
  background: var(--nr-accent-bg); color: var(--nr-accent); border:1px solid var(--nr-border); }}
.nr-badge.ok {{ background: var(--nr-ok-bg); color: var(--nr-ok); }}
.nr-card {{ border:1px solid var(--nr-border); border-radius:12px; padding:0.8rem 1rem;
  margin-bottom:0.6rem; background: var(--nr-bg); }}
:focus-visible {{ outline: 2px solid var(--nr-accent) !important; outline-offset: 2px; }}
/* Widgets use the same palette as the text, so no widget keeps a background that clashes. */
.stApp button {{ background-color: var(--nr-bg2) !important; color: var(--nr-text) !important;
  border-color: var(--nr-border) !important; }}
.stApp button * {{ color: inherit !important; }}
.stApp button[kind="primary"] {{ background-color: var(--nr-accent) !important;
  color: var(--nr-bg) !important; border-color: var(--nr-accent) !important; }}
.stApp button[aria-checked="true"] {{ background-color: var(--nr-accent-bg) !important;
  color: var(--nr-accent) !important; border-color: var(--nr-accent) !important; }}
.stApp input, .stApp textarea, .stApp [role="combobox"],
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stDateInput"] div[data-baseweb="input"],
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {{
  background-color: var(--nr-bg2) !important; color: var(--nr-text) !important;
  border-color: var(--nr-border) !important; }}
[data-baseweb="popover"] ul, [role="listbox"], [role="option"] {{
  background-color: var(--nr-bg2) !important; color: var(--nr-text) !important; }}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {{
  background-color: var(--nr-bg) !important; color: var(--nr-text) !important;
  border-color: var(--nr-border) !important; }}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: var(--nr-text) !important; }}
</style>"""
