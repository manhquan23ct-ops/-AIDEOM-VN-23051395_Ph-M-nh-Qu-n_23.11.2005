"""pages/home.py — Trang chủ AIDEOM-VN — Premium Redesign v2"""

import streamlit as st
import plotly.graph_objects as go
from data_loader import load_macro, load_sectors

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  (string concatenation — không dùng HTML comments)
# ─────────────────────────────────────────────────────────────
FONT_CSS = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900"
    "&family=Playfair+Display:ital,wght@0,700;0,900;1,700"
    "&family=JetBrains+Mono:wght@400;600&display=swap');"

    "html,body,[class*='css'],.stMarkdown,button{"
    "font-family:'Montserrat',sans-serif !important;}"

    ".stApp{background:#F6F9F7 !important;}"
    ".block-container{padding-top:0 !important;max-width:1140px !important;}"
    "#MainMenu,footer,header{visibility:hidden;}"

    "[data-testid='stMetric']{"
    "background:#ffffff;"
    "border:1px solid #DCE9E1;"
    "border-radius:14px;"
    "padding:1rem 1.3rem !important;}"

    "[data-testid='stMetricLabel']{"
    "font-family:'Montserrat',sans-serif !important;"
    "font-size:0.78rem !important;"
    "font-weight:700 !important;"
    "color:#6B8A7A !important;"
    "text-transform:uppercase;"
    "letter-spacing:0.06em;}"

    "[data-testid='stMetricValue']{"
    "font-family:'Montserrat',sans-serif !important;"
    "font-size:1.65rem !important;"
    "font-weight:800 !important;"
    "color:#1A2B1F !important;}"

    "[data-testid='stMetricDelta']{font-size:0.80rem !important;font-weight:700 !important;}"

    "[data-testid='stExpander']{"
    "border:1px solid #DCE9E1 !important;"
    "border-radius:14px !important;"
    "background:#ffffff !important;"
    "margin-bottom:0.5rem !important;}"

    ".streamlit-expanderHeader{"
    "font-family:'Montserrat',sans-serif !important;"
    "font-size:0.97rem !important;"
    "font-weight:700 !important;"
    "padding:0.85rem 1.2rem !important;}"

    "hr{border-color:#DCE9E1 !important;margin:1.6rem 0 !important;}"

    "[data-testid='stCaptionContainer']{font-size:0.78rem !important;color:#8AA898 !important;}"

    "@keyframes ticker{"
    "0%{transform:translateX(0);}100%{transform:translateX(-50%);}}"

    ".ticker-wrap{"
    "overflow:hidden;"
    "background:#163527;"
    "padding:9px 0;"
    "position:relative;"
    "left:50%;right:50%;"
    "margin-left:-50vw;margin-right:-50vw;"
    "width:100vw;}"

    ".ticker-inner{"
    "display:flex;"
    "width:max-content;"
    "animation:ticker 28s linear infinite;}"

    ".ticker-item{"
    "font-family:'JetBrains Mono',monospace;"
    "font-size:0.74rem;"
    "font-weight:500;"
    "color:rgba(255,255,255,0.72);"
    "white-space:nowrap;"
    "padding:0 2.8rem;"
    "letter-spacing:0.03em;}"

    ".ticker-item span{color:#6FCF97;margin-right:0.45rem;font-weight:700;}"

    "@keyframes pulse{"
    "0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(1.5);}}"

    ".pulse-dot{"
    "display:inline-block;"
    "width:7px;height:7px;"
    "background:#4CAF72;"
    "border-radius:50%;"
    "animation:pulse 2s ease-in-out infinite;"
    "margin-right:6px;"
    "vertical-align:middle;}"

    "@keyframes fadeUp{"
    "from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}"

    ".fade-up{animation:fadeUp 0.6s ease both;}"
    ".fade-up-2{animation:fadeUp 0.6s 0.12s ease both;}"
    ".fade-up-3{animation:fadeUp 0.6s 0.24s ease both;}"

    ".stat-num{"
    "background:linear-gradient(135deg,#1A6B3C 0%,#2E8B57 50%,#1A6B3C 100%);"
    "-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;"
    "background-clip:text;}"

    ".bento-grid{"
    "display:grid;"
    "grid-template-columns:repeat(2,minmax(0,1fr));"
    "gap:16px;"
    "animation:fadeUp .55s ease both;}"

    ".bento-panel{"
    "border-radius:20px;"
    "padding:1.15rem 1.1rem 1rem;"
    "position:relative;"
    "overflow:hidden;"
    "transition:transform 0.22s ease,box-shadow 0.22s ease;"
    "border-width:1.5px;"
    "border-style:solid;}"

    ".bento-panel:hover{"
    "transform:translateY(-4px);"
    "box-shadow:0 18px 44px rgba(15,35,60,.10);}"

    ".bento-wm{"
    "position:absolute;"
    "right:-6px;bottom:-20px;"
    "font-family:'Playfair Display',serif;"
    "font-size:7.5rem;"
    "font-weight:900;"
    "line-height:1;"
    "pointer-events:none;"
    "user-select:none;"
    "opacity:0.07;}"

    ".bento-head{"
    "display:flex;"
    "align-items:flex-start;"
    "justify-content:space-between;"
    "margin-bottom:.95rem;"
    "position:relative;"
    "z-index:1;}"

    ".diff-badge{"
    "display:inline-block;"
    "font-family:'JetBrains Mono',monospace;"
    "font-size:0.62rem;"
    "font-weight:800;"
    "letter-spacing:0.14em;"
    "text-transform:uppercase;"
    "padding:2px 9px;"
    "border-radius:20px;"
    "border-width:1px;"
    "border-style:solid;"
    "margin-bottom:4px;}"

    ".mini-card{"
    "background:rgba(255,255,255,0.82);"
    "border:1px solid rgba(255,255,255,0.88);"
    "border-radius:14px;"
    "padding:0.72rem 0.82rem;"
    "margin-bottom:0.58rem;"
    "position:relative;"
    "z-index:1;"
    "transition:transform 0.18s ease,background 0.18s ease;}"

    ".mini-card:hover{transform:translateX(4px);background:#fff;}"
    ".mini-card:last-child{margin-bottom:0;}"

    "@media(max-width:900px){.bento-grid{grid-template-columns:1fr;}}"
    "</style>"
)

CF = dict(family="Montserrat, sans-serif", size=12)

TICKER_ITEMS = [
    ("GDP 2025",       "514,0 tỷ USD  +8,02%"),
    ("Kinh tế số/GDP", "≈19,5%  +1,2dpt"),
    ("FDI giải ngân",  "27,6 tỷ USD  +8,9%"),
    ("GDP/người",      "5.026 USD  +6,9%"),
    ("Xuất khẩu 2025", "475,0 tỷ USD"),
    ("DN công nghệ số","80.052  +8,5%"),
    ("GII 2025",       "Hạng 44/139"),
    ("Đóng góp KH-CN", "2,49% GDP"),
    ("Startup AI",     "≈350+"),
    ("NSLĐ 2025",      "245 tr.VND/người"),
]


def _ticker_html():
    items = "".join(
        f'<span class="ticker-item"><span>▶</span>{k}: {v}</span>'
        for k, v in TICKER_ITEMS
    )
    return (
        '<div class="ticker-wrap">'
        '<div class="ticker-inner">'
        + items * 2 +
        '</div></div>'
    )


def _section_head(title: str, sub: str) -> str:
    """Trả về HTML cho header section nhất quán."""
    return (
        '<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:0.9rem;">'
        f'<span style="font-family:\'Playfair Display\',serif;font-size:1.3rem;'
        f'font-weight:700;color:#1A2B1F;">{title}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.73rem;'
        f'color:#8AA898;">{sub}</span>'
        '</div>'
    )


def render():
    st.markdown(FONT_CSS, unsafe_allow_html=True)
    macro   = load_macro()
    sectors = load_sectors()  # noqa: F841

    # ════════════════════════════════════════════════════════
    # LIVE TICKER
    # ════════════════════════════════════════════════════════
    st.markdown(_ticker_html(), unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # HERO
    # ════════════════════════════════════════════════════════
    hero_html = (
        '<div style="max-width:1140px;margin:0 auto;display:flex;align-items:center;'
        'justify-content:flex-start;gap:1.4rem;padding:2rem 0 1.8rem;flex-wrap:wrap;">'

        # Left column
        '<div style="flex:1;min-width:320px;">'
        '<div style="display:inline-flex;align-items:center;gap:7px;'
        'background:#EBF5EF;border:1px solid #C1DEC9;'
        'border-radius:20px;padding:4px 14px;margin-bottom:14px;">'
        '<span class="pulse-dot"></span>'
        '<span style="font-family:Montserrat,sans-serif;font-size:0.70rem;font-weight:700;'
        'color:#1A6B3C;letter-spacing:0.10em;text-transform:uppercase;">'
        'D\u1eef li\u1ec7u th\u1ef1c \u00b7 Vi\u1ec7t Nam 2020\u20132025</span></div>'

        '<div style="margin-bottom:8px;">'
        '<span style="font-family:JetBrains Mono,monospace;font-size:0.74rem;font-weight:600;'
        'color:#8AA898;letter-spacing:0.16em;text-transform:uppercase;display:block;margin-bottom:5px;">'
        '\U0001f1fb\U0001f1f3 &nbsp; Vietnam Economic Decision Model</span>'
        '<h1 style="font-family:Playfair Display,serif;font-size:3rem;font-weight:900;'
        'color:#1A2B1F;margin:0;line-height:1.05;letter-spacing:-1px;">'
        'AIDEOM<span style="font-style:italic;color:#2E8B57;">-VN</span></h1></div>'

        '<p style="font-family:Montserrat,sans-serif;font-size:0.94rem;font-weight:500;'
        'color:#4A6355;line-height:1.7;margin:0 0 16px;max-width:500px;">'
        'Web app gi\u1ea3i <b style="color:#1A6B3C;font-weight:800;">12 b\u00e0i to\u00e1n</b>'
        ' m\u00f4 h\u00ecnh ra quy\u1ebft \u0111\u1ecbnh ph\u00e1t tri\u1ec3n kinh t\u1ebf Vi\u1ec7t Nam'
        ' trong k\u1ec9 nguy\u00ean AI \u2014'
        ' t\u1eeb <b style="color:#1A6B3C;font-weight:800;">LP c\u01a1 b\u1ea3n</b>'
        ' \u0111\u1ebfn <b style="color:#1A6B3C;font-weight:800;">Q-learning</b></p>'

        '<div style="display:flex;gap:7px;flex-wrap:wrap;">'
        '<span style="background:#EBF5EF;color:#1A6B3C;font-size:0.75rem;font-weight:600;'
        'padding:5px 13px;border-radius:20px;border:1px solid #C1DEC9;">\U0001f40d Python 3.11</span>'
        '<span style="background:#EBF5EF;color:#1A6B3C;font-size:0.75rem;font-weight:600;'
        'padding:5px 13px;border-radius:20px;border:1px solid #C1DEC9;">\U0001f4ca NSO \u00b7 GSO \u00b7 WB</span>'
        '<span style="background:#EBF5EF;color:#1A6B3C;font-size:0.75rem;font-weight:600;'
        'padding:5px 13px;border-radius:20px;border:1px solid #C1DEC9;">\u2699\ufe0f PuLP \u00b7 CVXPY</span>'
        '<span style="background:#EBF5EF;color:#1A6B3C;font-size:0.75rem;font-weight:600;'
        'padding:5px 13px;border-radius:20px;border:1px solid #C1DEC9;">\U0001f916 Q-learning \u00b7 NSGA-II</span>'
        '</div></div>'

        # Right column — 2×2 stat grid
        '<div style="flex-shrink:0;">'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;'
        'background:#DCE9E1;border:1px solid #DCE9E1;border-radius:18px;overflow:hidden;">'

        '<div style="background:#fff;padding:1.2rem 1.6rem;text-align:center;">'
        '<div class="stat-num" style="font-family:Playfair Display,serif;font-size:2.4rem;'
        'font-weight:900;line-height:1;">12</div>'
        '<div style="font-size:0.74rem;font-weight:600;color:#8AA898;margin-top:4px;">'
        'B\u00e0i to\u00e1n</div></div>'

        '<div style="background:#fff;padding:1.2rem 1.6rem;text-align:center;">'
        '<div class="stat-num" style="font-family:Playfair Display,serif;font-size:2.4rem;'
        'font-weight:900;line-height:1;">4</div>'
        '<div style="font-size:0.74rem;font-weight:600;color:#8AA898;margin-top:4px;">'
        'C\u1ea5p \u0111\u1ed9</div></div>'

        '<div style="background:#fff;padding:1.2rem 1.6rem;text-align:center;">'
        '<div class="stat-num" style="font-family:Playfair Display,serif;font-size:2.4rem;'
        'font-weight:900;line-height:1;">3</div>'
        '<div style="font-size:0.74rem;font-weight:600;color:#8AA898;margin-top:4px;">'
        'File CSV</div></div>'

        '<div style="background:#EBF5EF;padding:1.2rem 1.6rem;text-align:center;">'
        '<div class="stat-num" style="font-family:Playfair Display,serif;font-size:2.4rem;'
        'font-weight:900;line-height:1;">6</div>'
        '<div style="font-size:0.74rem;font-weight:600;color:#1A6B3C;margin-top:4px;">'
        'N\u0103m d\u1eef li\u1ec7u</div></div>'

        '</div></div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════
    # KPI METRICS
    # ════════════════════════════════════════════════════════
    st.markdown(_section_head("Chỉ số kinh tế", "// Việt Nam 2025"), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("\U0001f4c8 GDP 2025",           "514,0 tỷ USD", "+8,02%")
    c2.metric("\U0001f4bb Kinh tế số / GDP",   "≈19,5%",       "+1,2 dpt")
    c3.metric("\U0001f3ed FDI giải ngân 2025", "27,6 tỷ USD",  "+8,9%")
    c4.metric("\U0001f464 GDP/người 2025",     "5.026 USD",    "+6,9%")

    st.divider()

    # ════════════════════════════════════════════════════════
    # CHARTS
    # ════════════════════════════════════════════════════════
    st.markdown(_section_head("Dữ liệu vĩ mô", "// 2020 → 2025"), unsafe_allow_html=True)

    CARD = (
        "background:#fff;border:1px solid #DCE9E1;"
        "border-radius:16px;padding:{p};"
    )

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown(
            '<div style="' + CARD.format(p="1.2rem 1.4rem 0.5rem") + '">'
            '<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:3px;">'
            '<span style="font-family:\'Playfair Display\',serif;font-size:1rem;'
            'font-weight:700;color:#1A2B1F;">T\u0103ng tr\u01b0\u1edfng GDP</span>'
            '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.70rem;'
            'color:#8AA898;">ngh\u00ecn t\u1ef7 VND \u00b7 %</span>'
            '</div>',
            unsafe_allow_html=True
        )
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=macro["year"], y=macro["GDP_trillion_VND"],
            marker_color=["#C8E6C9","#A5D6A7","#81C784","#4CAF72","#2E8B57","#1A6B3C"],
            marker_line_width=0,
            hovertemplate="%{x}: %{y:,.0f} ng.tỷ<extra></extra>",
        ))
        fig1.add_trace(go.Scatter(
            x=macro["year"], y=macro["GDP_growth_pct"],
            mode="lines+markers", yaxis="y2",
            line=dict(color="#FB8C00", width=2.5),
            marker=dict(size=7, color="#FB8C00", line=dict(width=2, color="#fff")),
        ))
        fig1.update_layout(
            height=220, margin=dict(l=0,r=0,t=5,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=CF, showlegend=False, bargap=0.28,
            yaxis=dict(showgrid=True, gridcolor="#EDF4EF", tickfont=dict(size=11), zeroline=False),
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        tickfont=dict(size=11), ticksuffix="%"),
            xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        # Chart 2: Digital economy
        st.markdown(
            '<div style="' + CARD.format(p="1rem 1.2rem 0.4rem") + 'margin-bottom:0.8rem;">'
            '<div style="display:flex;align-items:baseline;gap:6px;margin-bottom:2px;">'
            '<span style="font-family:\'Playfair Display\',serif;font-size:0.95rem;'
            'font-weight:700;color:#1A2B1F;">Kinh t\u1ebf s\u1ed1</span>'
            '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
            'color:#8AA898;">% GDP</span>'
            '</div>',
            unsafe_allow_html=True
        )
        fig2 = go.Figure(go.Scatter(
            x=macro["year"], y=macro["digital_economy_share_GDP_pct"],
            mode="lines+markers+text", fill="tozeroy",
            fillcolor="rgba(46,139,87,0.09)",
            line=dict(color="#2E8B57", width=2.5),
            marker=dict(size=7, color="#2E8B57", line=dict(width=2, color="#fff")),
            text=[f"{v}%" for v in macro["digital_economy_share_GDP_pct"]],
            textposition="top center", textfont=dict(size=9, color="#1A6B3C"),
        ))
        fig2.update_layout(
            height=100, margin=dict(l=0,r=0,t=8,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=CF,
            yaxis=dict(showgrid=True, gridcolor="#EDF4EF", ticksuffix="%", tickfont=dict(size=10)),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Chart 3: FDI
        st.markdown(
            '<div style="' + CARD.format(p="1rem 1.2rem 0.4rem") + '">'
            '<div style="display:flex;align-items:baseline;gap:6px;margin-bottom:2px;">'
            '<span style="font-family:\'Playfair Display\',serif;font-size:0.95rem;'
            'font-weight:700;color:#1A2B1F;">FDI gi\u1ea3i ng\u00e2n</span>'
            '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
            'color:#8AA898;">t\u1ef7 USD</span>'
            '</div>',
            unsafe_allow_html=True
        )
        fig3 = go.Figure(go.Bar(
            x=macro["year"], y=macro["FDI_disbursed_billion_USD"],
            marker_color="#1976D2", marker_line_width=0,
            text=[f"{v:.1f}" for v in macro["FDI_disbursed_billion_USD"]],
            textposition="outside", textfont=dict(size=9, color="#1976D2"),
        ))
        fig3.update_layout(
            height=97, margin=dict(l=0,r=0,t=8,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=CF, bargap=0.3,
            yaxis=dict(showgrid=True, gridcolor="#EDF4EF", tickfont=dict(size=10)),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════
    # TIẾN ĐỘ MỤC TIÊU 2030
    # ════════════════════════════════════════════════════════
    st.markdown(
        _section_head("Tiến độ mục tiêu 2030", "// NQ 57-NQ/TW · QĐ 749"),
        unsafe_allow_html=True
    )

    t1, t2, t3, t4, t5 = st.columns(5, gap="medium")
    targets = [
        (t1, "Kinh tế số/GDP", 19.5, 30,  "#1A6B3C"),
        (t2, "Tăng trưởng %",   8.0,  7.5, "#1976D2"),
        (t3, "FDI (tỷ USD)",   27.6, 40,  "#FB8C00"),
        (t4, "LĐ đào tạo %",   28.4, 40,  "#7B1FA2"),
        (t5, "DN công nghệ",   80.1, 100, "#E53935"),
    ]
    for col, label, cur, tgt, hex_color in targets:
        pct   = min(int(cur / tgt * 100), 100)
        r, g, b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
        badge = "✓ Đạt" if pct >= 100 else (f"{pct}%" if pct >= 70 else f"{pct}%")
        with col:
            st.markdown(
                '<div style="background:#fff;border:1px solid #DCE9E1;border-radius:14px;'
                'padding:0.9rem 1rem;text-align:center;">'
                f'<div style="font-size:0.70rem;font-weight:700;color:#6B8A7A;'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:7px;">{label}</div>'
                f'<div style="font-family:\'Playfair Display\',serif;font-size:1.8rem;'
                f'font-weight:900;color:{hex_color};line-height:1;margin-bottom:8px;">{badge}</div>'
                f'<div style="background:#EDF1EE;border-radius:20px;height:5px;'
                f'overflow:hidden;margin-bottom:6px;">'
                f'<div style="width:{pct}%;height:100%;border-radius:20px;'
                f'background:linear-gradient(90deg,rgba({r},{g},{b},.55),rgba({r},{g},{b},1));"></div>'
                f'</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
                f'color:#8AA898;">{cur} / {tgt}</div>'
                '</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ════════════════════════════════════════════════════════
    # 12 BÀI TOÁN — BENTO BOARD
    # ════════════════════════════════════════════════════════
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:1rem;">'
        '<span style="font-family:\'Playfair Display\',serif;font-size:1.36rem;'
        'font-weight:800;color:#1A2B1F;">12 B\u00e0i to\u00e1n</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.73rem;'
        'color:#8AA898;">// M\u00f4 h\u00ecnh Ra Quy\u1ebft \u0111\u1ecbnh</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # cap, hex_color, bg, border, watermark_range, items
    levels = [
        ("DỄ",        "#1A6B3C", "#EBF5EF", "#BFD9C8", "01–03", [
            ("01", "Cobb-Douglas + AI",  "Growth accounting, dự báo GDP 2030",   "numpy · pandas"),
            ("02", "LP ngân sách số",    "Tối đa GDP kỳ vọng — shadow price",     "scipy · pulp"),
            ("03", "Priority 10 ngành",  "Min-max norm, sensitivity analysis",    "pandas · seaborn"),
        ]),
        ("TRUNG BÌNH","#C05000", "#FFF7ED", "#FDDBA8", "04–06", [
            ("04", "LP ngành-vùng",     "24 biến, ràng buộc công bằng vùng",     "pulp · cvxpy"),
            ("05", "MIP 15 dự án",      "Biến nhị phân, tiên quyết, loại trừ",   "pulp · CBC"),
            ("06", "TOPSIS 6 vùng",     "Entropy weights khách quan",             "numpy · sklearn"),
        ]),
        ("KHÁ KHÓ",  "#B91C1C", "#FEF2F2", "#FECACA", "07–09", [
            ("07", "NSGA-II Pareto",    "4 mục tiêu xung đột, biên Pareto 3D",   "pymoo · plotly"),
            ("08", "Tối ưu động",       "2026–2035, Bellman, phân tích cú sốc",  "cvxpy · SLSQP"),
            ("09", "Lao động & AI",     "NetJob ròng, Sankey, ngưỡng đào tạo",   "cvxpy · pulp"),
        ]),
        ("KHÓ",       "#3730A3", "#F5F3FF", "#C4B5FD", "10–12", [
            ("10", "Stochastic SP",     "VSS · EVPI · 4 kịch bản toàn cầu",      "pyomo · glpk"),
            ("11", "Q-learning RL",     "MDP 81 trạng thái, 5 hành động",        "gymnasium · SB3"),
            ("12", "AIDEOM tích hợp",  "6 module · dashboard · 5 kịch bản",     "streamlit · all"),
        ]),
    ]

    panels = ""
    for cap, hex_color, bg, border_c, wm, items in levels:
        cards = ""
        for num, title, desc, tools in items:
            cards += (
                '<div class="mini-card">'
                '<div style="display:flex;gap:9px;align-items:flex-start;">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.76rem;'
                f'font-weight:800;color:{hex_color};background:{bg};'
                f'border:1px solid {border_c};border-radius:10px;'
                f'min-width:36px;text-align:center;padding:5px 0;flex-shrink:0;">{num}</div>'
                '<div style="flex:1;min-width:0;">'
                f'<div style="font-weight:800;color:#172334;font-size:.89rem;'
                f'line-height:1.25;margin-bottom:2px;">{title}</div>'
                f'<div style="font-size:.77rem;color:#53695B;line-height:1.4;'
                f'margin-bottom:4px;">{desc}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.63rem;'
                f'font-weight:700;color:{hex_color};opacity:.8;">{tools}</div>'
                '</div></div></div>'
            )

        panels += (
            f'<div class="bento-panel" style="background:{bg};border-color:{border_c};">'
            f'<div class="bento-wm" style="color:{hex_color};">{wm}</div>'
            '<div class="bento-head">'
            '<div>'
            f'<div class="diff-badge" style="color:{hex_color};'
            f'background:rgba(255,255,255,0.75);border-color:{border_c};">CẤP ĐỘ</div>'
            f'<div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;'
            f'font-weight:900;color:{hex_color};line-height:1.05;">{cap}</div>'
            '</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
            f'font-weight:600;color:{hex_color};opacity:0.45;letter-spacing:0.05em;'
            f'margin-top:2px;">B\u00e0i {wm}</div>'
            '</div>'
            f'{cards}'
            '</div>'
        )

    st.markdown(f'<div class="bento-grid">{panels}</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("📚 Nguồn: NSO/GSO 2026 · MoST 2026 · MPI · World Bank · WIPO GII 2025")
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
