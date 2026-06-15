"""
utils.py
--------
Các hàm UI/helper dùng chung cho toàn bộ pages AIDEOM-VN.
Import: from utils import bai_header, info_box, card, coming_soon, STYLE
"""

import streamlit as st
import random

# ── Màu cấp độ ────────────────────────────────────────────────────────────────
LEVEL_COLOR = {
    "DỄ":      "#4CAF72",
    "TB":       "#FB8C00",
    "KHÁ KHÓ": "#E53935",
    "KHÓ":     "#7B1FA2",
}

# ── CSS toàn cục (gọi 1 lần trong app.py) ────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:        #F4F6F8;
    --surface:   #FFFFFF;
    --primary:   #1A6B3C;
    --primary2:  #2E8B57;
    --accent:    #4CAF72;
    --text:      #1A2B1F;
    --text2:     #4A6355;
    --text3:     #8AA898;
    --border:    #DDE8E2;
    --radius:    14px;
    --shadow:    0 2px 12px rgba(26,107,60,0.08);
}

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container,
section.main > div.block-container {
    padding: 1.2rem 1.5rem 2.5rem 1.5rem !important;
    max-width: 1800px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
[data-testid="stAppViewContainer"] > section.main {
    background-color: #F4F6F8 !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="collapsedControl"],
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebar"] > div > div > button {
    display: none !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1.5px solid var(--border) !important;
    box-shadow: 2px 0 16px rgba(26,107,60,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

.stButton > button {
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--primary2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(26,107,60,0.25) !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricLabel"]  { font-size: 0.78rem !important; color: var(--text2) !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"]  { font-size: 1.9rem !important; font-weight: 800 !important; color: var(--primary) !important; }

[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #81C784; border-radius: 4px; }
</style>
"""


# ── Component: Header banner cho từng bài ────────────────────────────────────
def bai_header(so: str, ten: str, mo_ta: str,
               cap_do: str, tools: list, thoi_luong: str):
    cap_color = LEVEL_COLOR.get(cap_do, "#888")
    tools_html = "".join([
        '<span style="background:rgba(255,255,255,0.2);color:#fff;font-size:0.72rem;'
        'font-weight:600;padding:2px 10px;border-radius:20px;">' + t + '</span>'
        for t in tools
    ])
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1A6B3C 0%,#2E8B57 60%,#4CAF72 100%);'
        'border-radius:0;padding:2.4rem 2.5rem 2rem;color:#fff;'
        'box-shadow:0 4px 24px rgba(26,107,60,0.2);">'
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap;">'
        '<span style="background:rgba(255,255,255,0.2);font-size:0.72rem;font-weight:700;'
        'padding:3px 12px;border-radius:20px;letter-spacing:0.05em;">BÀI ' + so + '</span>'
        '<span style="background:' + cap_color + ';font-size:0.72rem;font-weight:700;'
        'padding:3px 12px;border-radius:20px;">' + cap_do + '</span>'
        '<span style="background:rgba(255,255,255,0.15);font-size:0.72rem;'
        'padding:3px 10px;border-radius:20px;">⏱ ' + thoi_luong + '</span>'
        '</div>'
        '<h2 style="margin:0 0 8px;font-size:1.6rem;font-weight:800;line-height:1.2;">' + ten + '</h2>'
        '<p style="margin:0 0 16px;opacity:0.85;font-size:0.9rem;line-height:1.6;">' + mo_ta + '</p>'
        '<div style="display:flex;gap:6px;flex-wrap:wrap;">' + tools_html + '</div>'
        '</div>'
        '<div style="padding:0 2.5rem;">',
        unsafe_allow_html=True
    )


def end_padding():
    """Đóng div padding sau bai_header"""
    st.markdown("</div>", unsafe_allow_html=True)


# ── Component: Info box màu ───────────────────────────────────────────────────
def info_box(text: str, bg="#E8F5E9", border="#4CAF72", icon="💡"):
    st.markdown(
        '<div style="background:' + bg + ';border-left:4px solid ' + border + ';border-radius:10px;'
        'padding:0.9rem 1.2rem;margin:0.8rem 0;font-size:0.87rem;'
        'color:var(--text);line-height:1.6;">'
        + icon + ' ' + text +
        '</div>',
        unsafe_allow_html=True
    )


# ── Component: Metric card tùy chỉnh ─────────────────────────────────────────
def card(title: str, value: str, delta=None,
         delta_label="", icon="", color="var(--primary)") -> str:
    delta_html = ""
    if delta is not None:
        d_color = "#2E8B57" if delta >= 0 else "#E53935"
        d_arrow = "▲" if delta >= 0 else "▼"
        delta_html = (
            '<div style="font-size:0.78rem;color:' + d_color + ';'
            'margin-top:4px;font-weight:600;">'
            + d_arrow + ' ' + str(abs(delta)) + ' ' + delta_label +
            '</div>'
        )
    return (
        '<div style="background:var(--surface);border:1.5px solid var(--border);'
        'border-radius:var(--radius);padding:1.2rem 1.4rem;'
        'box-shadow:var(--shadow);height:100%;">'
        '<div style="font-size:0.72rem;color:var(--text2);font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">'
        + icon + ' ' + title +
        '</div>'
        '<div style="font-size:2rem;font-weight:800;color:' + color + ';line-height:1.1;">' + value + '</div>'
        + delta_html +
        '</div>'
    )


# ── Component: Placeholder "đang phát triển" ─────────────────────────────────
def coming_soon(msg="Phần code giải bài sẽ được bổ sung sau"):
    st.markdown(
        '<div style="text-align:center;padding:3.5rem 2rem;background:var(--surface);'
        'border:2px dashed var(--border);border-radius:16px;margin:1.5rem 0;">'
        '<div style="font-size:2.8rem;margin-bottom:0.8rem;">🚧</div>'
        '<h3 style="color:var(--text2);font-weight:700;margin:0 0 6px;">Đang phát triển</h3>'
        '<p style="color:var(--text3);font-size:0.88rem;margin:0;">' + msg + '</p>'
        '</div>',
        unsafe_allow_html=True
    )


# ── Component: Section header ─────────────────────────────────────────────────
def section_title(title: str, emoji=""):
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;'
        'margin:1.4rem 0 0.8rem;padding-bottom:0.5rem;'
        'border-bottom:2px solid var(--border);">'
        '<span style="font-size:1.1rem;">' + emoji + '</span>'
        '<span style="font-weight:700;font-size:1rem;color:var(--text);">' + title + '</span>'
        '</div>',
        unsafe_allow_html=True
    )


# ── Component: Code snippet ───────────────────────────────────────────────────
def code_hint(code: str):
    st.markdown(
        '<div style="background:#1E1E1E;border-radius:10px;padding:1rem 1.2rem;'
        'margin:0.5rem 0;overflow-x:auto;">'
        '<pre style="margin:0;font-family:\'JetBrains Mono\',monospace;'
        'font-size:0.8rem;color:#D4D4D4;line-height:1.6;">' + code + '</pre>'
        '</div>',
        unsafe_allow_html=True
    )


# ── Loading animations ────────────────────────────────────────────────────────
def splash_loader():
    """Full-screen splash — tự tắt qua CSS animation sau 3s, không dùng JS timer."""
    if "splashed" in st.session_state:
        return
    st.session_state["splashed"] = True
    variant = random.choice(["bars", "rings", "dots"])

    css = (
        "<style id='sp-style'>"
        "@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&display=swap');"
        "#sp{"
        "position:fixed;top:0;left:0;width:100vw;height:100vh;"
        "background:#1A2B1F;z-index:2147483647;"
        "display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;gap:24px;"
        "animation:sp-out 0.45s ease 3s forwards;"
        "}"
        "@keyframes sp-out{to{opacity:0;visibility:hidden;pointer-events:none}}"
        ".sp-name{font-family:'Montserrat',sans-serif;font-size:56px;font-weight:800;"
        "color:#fff;letter-spacing:.09em;text-align:center;"
        "animation:sp-fi .7s ease .15s both;}"
        ".sp-name span{color:#4CAF72;}"
        ".sp-sub{font-family:'JetBrains Mono',monospace;font-size:17px;"
        "color:#fff;letter-spacing:.22em;text-align:center;"
        "margin-top:-14px;"
        "animation:sp-fi .7s ease .35s both;}"
        ".sp-track{width:180px;height:2px;background:#1f3328;"
        "border-radius:1px;overflow:hidden;margin-top:6px;"
        "animation:sp-fi .7s ease .5s both;}"
        ".sp-swipe{height:100%;width:36%;background:#4CAF72;border-radius:1px;"
        "animation:sp-sw 1.9s ease-in-out infinite;}"
        "@keyframes sp-sw{0%{transform:translateX(-110%)}100%{transform:translateX(390%)}}"
        "@keyframes sp-fi{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}"
        ".bw{display:flex;align-items:flex-end;gap:5px;height:64px;animation:sp-fi .5s ease both;}"
        ".b{width:9px;border-radius:3px 3px 0 0;"
        "animation:rb 1.4s ease-in-out infinite;transform-origin:bottom center;}"
        "@keyframes rb{0%,100%{transform:scaleY(.08);opacity:.15}50%{transform:scaleY(1);opacity:1}}"
        ".rw{position:relative;width:68px;height:68px;"
        "display:flex;align-items:center;justify-content:center;"
        "animation:sp-fi .5s ease both;}"
        ".rc{width:16px;height:16px;border-radius:50%;background:#4CAF72;"
        "z-index:1;position:relative;}"
        ".ri{position:absolute;border:1.5px solid #4CAF72;border-radius:50%;"
        "width:24px;height:24px;opacity:0;animation:rp 2.1s ease-out infinite;}"
        "@keyframes rp{0%{transform:scale(1);opacity:.9}100%{transform:scale(3.5);opacity:0}}"
        ".dw{display:flex;gap:11px;align-items:center;animation:sp-fi .5s ease both;}"
        ".d{width:13px;height:13px;border-radius:50%;animation:db 1.15s ease-in-out infinite;}"
        "@keyframes db{0%,100%{transform:translateY(0);opacity:.2}50%{transform:translateY(-22px);opacity:1}}"
        "</style>"
    )

    if variant == "bars":
        anim_html = (
            "<div class='bw'>"
            "<div class='b' style='height:36px;background:#4CAF72;animation-delay:0s'></div>"
            "<div class='b' style='height:50px;background:#2E8B57;animation-delay:.1s'></div>"
            "<div class='b' style='height:64px;background:#4CAF72;animation-delay:.2s'></div>"
            "<div class='b' style='height:42px;background:#1A6B3C;animation-delay:.3s'></div>"
            "<div class='b' style='height:56px;background:#2E8B57;animation-delay:.4s'></div>"
            "<div class='b' style='height:38px;background:#4CAF72;animation-delay:.5s'></div>"
            "</div>"
        )
    elif variant == "rings":
        anim_html = (
            "<div class='rw'>"
            "<div class='rc'></div>"
            "<div class='ri' style='animation-delay:0s'></div>"
            "<div class='ri' style='animation-delay:.7s'></div>"
            "<div class='ri' style='animation-delay:1.4s'></div>"
            "</div>"
        )
    else:
        anim_html = (
            "<div class='dw'>"
            "<div class='d' style='background:#4CAF72;animation-delay:0s'></div>"
            "<div class='d' style='background:#2E8B57;animation-delay:.16s'></div>"
            "<div class='d' style='background:#1A6B3C;animation-delay:.32s'></div>"
            "</div>"
        )

    overlay = (
        "<div id='sp'>"
        + anim_html
        + "<div class='sp-name'>AIDEOM<span>-VN</span></div>"
        + "<div class='sp-sub'>PHÂN TÍCH KINH TẾ VIỆT NAM</div>"
        + "<div class='sp-track'><div class='sp-swipe'></div></div>"
        + "</div>"
    )

    # JS chỉ dùng để: move lên body (full screen) + dọn DOM sau 4s
    js = (
        "<script>"
        "(function(){"
        "var el=document.getElementById('sp');"
        "if(el)document.body.appendChild(el);"
        "setTimeout(function(){"
        "var sp=document.getElementById('sp');"
        "if(sp)sp.remove();"
        "var st=document.getElementById('sp-style');"
        "if(st)st.remove();"
        "},4000);"
        "})();"
        "</script>"
    )

    st.markdown(css + overlay + js, unsafe_allow_html=True)

    css = (
        "<style id='sp-style'>"
        "@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&display=swap');"
        "#sp{"
        "position:fixed;top:0;left:0;width:100vw;height:100vh;"
        "background:#1A2B1F;"
        "z-index:2147483647;"
        "display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;gap:24px;"
        "}"
        ".sp-name{"
        "font-family:'Montserrat',sans-serif;"
        "font-size:56px;font-weight:800;"
        "color:#fff;letter-spacing:.09em;text-align:center;"
        "animation:sp-fi .7s ease .15s both;"
        "}"
        ".sp-name span{color:#4CAF72;}"
        ".sp-sub{"
        "font-family:'JetBrains Mono',monospace;"
        "font-size:17px;color:#fff;"
        "letter-spacing:.22em;text-align:center;"
        "margin-top:-14px;"
        "animation:sp-fi .7s ease .35s both;"
        "}"
        ".sp-track{"
        "width:180px;height:2px;background:#1f3328;"
        "border-radius:1px;overflow:hidden;margin-top:6px;"
        "animation:sp-fi .7s ease .5s both;"
        "}"
        ".sp-swipe{"
        "height:100%;width:36%;background:#4CAF72;"
        "border-radius:1px;"
        "animation:sp-sw 1.9s ease-in-out infinite;"
        "}"
        "@keyframes sp-sw{0%{transform:translateX(-110%)}100%{transform:translateX(390%)}}"
        "@keyframes sp-fi{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}"
        ".bw{display:flex;align-items:flex-end;gap:5px;height:64px;animation:sp-fi .5s ease both;}"
        ".b{width:9px;border-radius:3px 3px 0 0;"
        "animation:rb 1.4s ease-in-out infinite;transform-origin:bottom center;}"
        "@keyframes rb{0%,100%{transform:scaleY(.08);opacity:.15}50%{transform:scaleY(1);opacity:1}}"
        ".rw{position:relative;width:68px;height:68px;"
        "display:flex;align-items:center;justify-content:center;"
        "animation:sp-fi .5s ease both;}"
        ".rc{width:16px;height:16px;border-radius:50%;background:#4CAF72;"
        "z-index:1;position:relative;}"
        ".ri{position:absolute;border:1.5px solid #4CAF72;border-radius:50%;"
        "width:24px;height:24px;opacity:0;"
        "animation:rp 2.1s ease-out infinite;}"
        "@keyframes rp{0%{transform:scale(1);opacity:.9}100%{transform:scale(3.5);opacity:0}}"
        ".dw{display:flex;gap:11px;align-items:center;animation:sp-fi .5s ease both;}"
        ".d{width:13px;height:13px;border-radius:50%;"
        "animation:db 1.15s ease-in-out infinite;}"
        "@keyframes db{0%,100%{transform:translateY(0);opacity:.2}50%{transform:translateY(-22px);opacity:1}}"
        "</style>"
    )

    if variant == "bars":
        anim_html = (
            "<div class='bw'>"
            "<div class='b' style='height:36px;background:#4CAF72;animation-delay:0s'></div>"
            "<div class='b' style='height:50px;background:#2E8B57;animation-delay:.1s'></div>"
            "<div class='b' style='height:64px;background:#4CAF72;animation-delay:.2s'></div>"
            "<div class='b' style='height:42px;background:#1A6B3C;animation-delay:.3s'></div>"
            "<div class='b' style='height:56px;background:#2E8B57;animation-delay:.4s'></div>"
            "<div class='b' style='height:38px;background:#4CAF72;animation-delay:.5s'></div>"
            "</div>"
        )
    elif variant == "rings":
        anim_html = (
            "<div class='rw'>"
            "<div class='rc'></div>"
            "<div class='ri' style='animation-delay:0s'></div>"
            "<div class='ri' style='animation-delay:.7s'></div>"
            "<div class='ri' style='animation-delay:1.4s'></div>"
            "</div>"
        )
    else:
        anim_html = (
            "<div class='dw'>"
            "<div class='d' style='background:#4CAF72;animation-delay:0s'></div>"
            "<div class='d' style='background:#2E8B57;animation-delay:.16s'></div>"
            "<div class='d' style='background:#1A6B3C;animation-delay:.32s'></div>"
            "</div>"
        )

    overlay = (
        "<div id='sp'>"
        + anim_html
        + "<div class='sp-name'>AIDEOM<span>-VN</span></div>"
        + "<div class='sp-sub'>PHÂN TÍCH KINH TẾ VIỆT NAM</div>"
        + "<div class='sp-track'><div class='sp-swipe'></div></div>"
        + "</div>"
    )

    # JS: IIFE chạy ngay — move #sp vào body rồi dismiss sau đúng 3s
    js = (
        "<script>"
        "(function(){"
        "var el=document.getElementById('sp');"
        "if(el)document.body.appendChild(el);"
        "setTimeout(function(){"
        "var sp=document.getElementById('sp');"
        "if(!sp)return;"
        "sp.style.transition='opacity 0.4s ease';"
        "sp.style.opacity='0';"
        "setTimeout(function(){"
        "var sp2=document.getElementById('sp');"
        "if(sp2)sp2.remove();"
        "var st=document.getElementById('sp-style');"
        "if(st)st.remove();"
        "},450);"
        "},3000);"
        "})();"
        "</script>"
    )

    st.markdown(css + overlay + js, unsafe_allow_html=True)


def page_transition():
    """Bar xanh chạy ngang đầu trang — gọi đầu mỗi render()."""
    st.markdown(
        "<style>"
        "#ptb{position:fixed;top:0;left:0;height:3px;background:#2E8B57;"
        "border-radius:0 2px 2px 0;z-index:9999;"
        "animation:ptf .8s ease-out forwards;}"
        "@keyframes ptf{0%{width:0}35%{width:56%}80%{width:92%}100%{width:100%}}"
        "</style>"
        "<div id='ptb'></div>"
        "<script>"
        "setTimeout(function(){"
        "var e=document.getElementById('ptb');"
        "if(e){e.style.opacity='0';e.style.transition='opacity .3s';"
        "setTimeout(function(){if(e&&e.parentNode)e.parentNode.removeChild(e);},300);}"
        "},950);"
        "</script>",
        unsafe_allow_html=True
    )
