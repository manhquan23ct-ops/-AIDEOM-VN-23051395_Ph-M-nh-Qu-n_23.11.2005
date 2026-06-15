"""
app.py
------
Entry point của AIDEOM-VN.
Chỉ chứa: cấu hình trang, sidebar navigation, router.
KHÔNG chứa logic bài — mỗi bài nằm trong pages/baiX.py
"""

import importlib
import streamlit as st
import pandas as pd
from utils import GLOBAL_CSS, splash_loader, page_transition

# ── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIDEOM-VN | Mô hình ra quyết định",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global styles + splash (thứ tự quan trọng)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
splash_loader()

# Ẩn sidebar toggle + nav mặc định của Streamlit
st.markdown(
    "<style>"
    "[data-testid='stSidebarNav']{display:none!important;}"
    "[data-testid='collapsedControl']{display:none!important;}"
    "[data-testid='stSidebarCollapsedControl']{display:none!important;}"
    "[data-testid='stSidebarCollapseButton']{display:none!important;}"
    "button[aria-label='Close sidebar']{display:none!important;}"
    "section[data-testid='stSidebar']{"
    "min-width:400px!important;max-width:400px!important;"
    "width:400px!important;transform:translateX(0px)!important;"
    "visibility:visible!important;display:block!important;}"
    "section[data-testid='stSidebar'][aria-expanded='false']{"
    "transform:translateX(0px)!important;"
    "visibility:visible!important;display:block!important;}"
    "</style>",
    unsafe_allow_html=True
)

# ── Danh sách pages ───────────────────────────────────────────────────────────
PAGES = {
    "home":  ("🏠 Trang chủ",                "",          "pages.home"),
    "bai1":  ("📚 Bài 1 — Cobb-Douglas+AI",  "DỄ",        "pages.bai1"),
    "bai2":  ("💰 Bài 2 — LP ngân sách số",  "DỄ",        "pages.bai2"),
    "bai3":  ("🏆 Bài 3 — Priority 10 ngành","DỄ",        "pages.bai3"),
    "bai4":  ("🗺️ Bài 4 — LP ngành-vùng",   "TB",        "pages.bai4"),
    "bai5":  ("🎯 Bài 5 — MIP 15 dự án",    "TB",        "pages.bai5"),
    "bai6":  ("🌐 Bài 6 — TOPSIS 6 vùng",   "TB",        "pages.bai6"),
    "bai7":  ("🔴 Bài 7 — NSGA-II Pareto",  "KHÁ KHÓ",   "pages.bai7"),
    "bai8":  ("⏳ Bài 8 — Động 2026-2035",  "KHÁ KHÓ",   "pages.bai8"),
    "bai9":  ("👥 Bài 9 — Lao động & AI",   "KHÁ KHÓ",   "pages.bai9"),
    "bai10": ("🎲 Bài 10 — Stochastic SP",  "KHÓ",        "pages.bai10"),
    "bai11": ("🤖 Bài 11 — Q-learning RL",  "KHÓ",        "pages.bai11"),
    "bai12": ("🇻🇳 Bài 12 — AIDEOM tích hợp","KHÓ",      "pages.bai12"),
}

LEVEL_COLOR = {
    "DỄ": "#4CAF72", "TB": "#FB8C00",
    "KHÁ KHÓ": "#E53935", "KHÓ": "#7B1FA2",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(
        "<div style='padding:1.4rem 1.2rem 1rem;border-bottom:1.5px solid var(--border);'>"
        "<div style='display:flex;align-items:center;gap:10px;'>"
        "<div style='width:38px;height:38px;"
        "background:linear-gradient(135deg,#1A6B3C,#4CAF72);"
        "border-radius:10px;display:flex;align-items:center;"
        "justify-content:center;font-size:1.2rem;'>🇻🇳</div>"
        "<div>"
        "<div style='font-weight:800;font-size:1rem;color:#1A2B1F;'>AIDEOM-VN</div>"
        "<div style='font-size:0.67rem;color:#8AA898;font-weight:500;'>"
        "Decision Optimization Model</div>"
        "</div></div></div>",
        unsafe_allow_html=True
    )

    # ── Project stats strip ───────────────────────────────────────────────────
    st.markdown(
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);"
        "gap:6px;padding:0.9rem 1rem;border-bottom:1.5px solid var(--border);'>"
        + "".join([
            "<div style='background:#F4F6F8;border-radius:10px;"
            "padding:0.55rem 0.4rem;text-align:center;'>"
            "<div style='font-size:1.1rem;font-weight:800;color:#1A6B3C;line-height:1;'>"
            + v + "</div>"
            "<div style='font-size:0.6rem;color:#8AA898;font-weight:600;"
            "margin-top:3px;letter-spacing:0.04em;'>" + l + "</div>"
            "</div>"
            for v, l in [("12", "BÀI TOÁN"), ("6", "MODULE"), ("3", "CSV")]
        ])
        + "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='padding:0.8rem 1rem 0.2rem;font-size:0.67rem;"
        "font-weight:700;color:#8AA898;letter-spacing:0.1em;'>ĐIỀU HƯỚNG</div>",
        unsafe_allow_html=True
    )

    if "page" not in st.session_state:
        st.session_state.page = "home"

    for key, (label, level, _) in PAGES.items():
        is_active = st.session_state.page == key
        lc     = LEVEL_COLOR.get(level, "transparent")
        badge  = (
            "<span style='font-size:0.6rem;font-weight:700;padding:1px 6px;"
            "border-radius:20px;background:" + lc + ";color:#fff;'>" + level + "</span>"
            if level else ""
        )
        bg     = "linear-gradient(90deg,#E8F5E9,#F1F8F2)" if is_active else "transparent"
        border = "2px solid #1A6B3C" if is_active else "2px solid transparent"
        fw     = "700" if is_active else "500"
        col    = "#1A6B3C" if is_active else "#4A6355"

        st.markdown(
            "<div style='display:flex;align-items:center;justify-content:space-between;"
            "padding:0.42rem 1rem;margin:1px 0.4rem;border-radius:10px;"
            "background:" + bg + ";border-left:" + border + ";'>"
            "<span style='font-size:0.82rem;font-weight:" + fw + ";color:" + col + ";'>" + label + "</span>"
            + badge +
            "</div>",
            unsafe_allow_html=True
        )

        if st.button(label, key="nav_" + key, use_container_width=True):
            st.session_state.page = key
            st.rerun()

    # ── Vietnam macro snapshot ────────────────────────────────────────────────
    try:
        _df = pd.read_csv("data/vietnam_macro_2020_2025.csv")
        _r   = _df[_df["year"] == 2025].iloc[0]
        _gdp_g  = f"{_r['GDP_growth_pct']:.2f}%"
        _gdp_pc = f"${int(_r['GDP_per_capita_USD']):,}"
        _exp    = f"${_r['exports_billion_USD']:.1f}B"
        _macros = [
            ("📈 GDP tăng trưởng", _gdp_g,  "#1A6B3C"),
            ("💵 GDP/người",       _gdp_pc, "#2E8B57"),
            ("📦 Xuất khẩu",      _exp,    "#4CAF72"),
        ]
        _macro_html = (
            "<div style='padding:0.7rem 1rem 0.5rem;"
            "border-top:1.5px solid var(--border);margin-bottom:56px;'>"
            "<div style='font-size:0.67rem;font-weight:700;color:#8AA898;"
            "letter-spacing:0.1em;margin-bottom:8px;'>DỮ LIỆU 2025</div>"
        )
        for lbl, val, col in _macros:
            _macro_html += (
                "<div style='display:flex;justify-content:space-between;"
                "align-items:center;padding:4px 0;border-bottom:1px solid #F0F4F0;'>"
                "<span style='font-size:0.75rem;color:#4A6355;'>" + lbl + "</span>"
                "<span style='font-size:0.8rem;font-weight:700;color:" + col + ";'>"
                + val + "</span></div>"
            )
        _macro_html += "</div>"
        st.markdown(_macro_html, unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='margin-bottom:56px'></div>", unsafe_allow_html=True)

    # Footer sidebar
    st.markdown(
        "<div style='position:fixed;bottom:0;width:242px;padding:0.9rem 1.2rem;"
        "background:var(--surface);border-top:1.5px solid var(--border);'>"
        "<div style='font-size:0.7rem;color:#8AA898;'>📊 NSO · MoST · MIC · WB · GII 2025</div>"
        "<div style='font-size:0.7rem;color:#8AA898;margin-top:2px;'>"
        "🎓 Môn: Mô hình ra quyết định</div>"
        "</div>",
        unsafe_allow_html=True
    )

# ── Router: import động page tương ứng ───────────────────────────────────────
current_page = st.session_state.get("page", "home")
module_path  = PAGES[current_page][2]

try:
    mod = importlib.import_module(module_path)
    page_transition()   # bar xanh chạy ngang khi chuyển trang
    mod.render()
except ModuleNotFoundError:
    st.error("⚠️ Chưa tìm thấy file " + module_path.replace(".", "/") + ".py")
except Exception as e:
    st.exception(e)
