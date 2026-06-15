"""pages/bai6.py — Bài 6: TOPSIS xếp hạng 6 vùng kinh tế theo ưu tiên đầu tư AI

Bản làm lại:
- Cài TOPSIS từ đầu bằng numpy.
- Tính trọng số chuyên gia và trọng số Entropy theo đúng gợi ý đề.
- Phân tích độ nhạy w_AI từ 0.10 đến 0.40.
- Bổ sung AHP đơn giản cho câu 6.4.4 thay vì chỉ vẽ radar.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from utils import bai_header, end_padding, info_box, section_title
except Exception:  # fallback để file vẫn chạy độc lập khi thiếu utils.py
    def bai_header(so, ten, mo_ta, cap_do, tools, thoi_luong):
        st.title(f"Bài {so}. {ten}")
        st.caption(mo_ta)
    def end_padding():
        st.markdown("<br><br>", unsafe_allow_html=True)
    def info_box(text, bg="#F4F6F8", border="#1A6B3C", icon="ℹ️"):
        st.markdown(
            f"""
            <div style="background:{bg};border-left:4px solid {border};border-radius:10px;
                        padding:0.9rem 1rem;margin:0.7rem 0;color:#1A2B1F;">
                <span style="font-size:1.1rem;">{icon}</span>&nbsp;{text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    def section_title(text, icon=""):
        st.markdown(f"### {icon} {text}")


# ── Style ───────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"; C2 = "#2E8B57"; C3 = "#4CAF72"
CBLUE = "#1976D2"; CRED = "#E53935"; CORANGE = "#E65100"; CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

# ── Cấu hình mô hình ─────────────────────────────────────────────────────────
CRITERIA = [
    "grdp_per_capita_million_VND",
    "fdi_registered_billion_USD",
    "digital_index_0_100",
    "ai_readiness_0_100",
    "trained_labor_pct",
    "rd_intensity_pct",
    "internet_penetration_pct",
    "gini_coef",
]
CRITERIA_VI = [
    "GRDP/người",
    "FDI",
    "Digital Index",
    "AI Readiness",
    "LĐ đào tạo",
    "R&D/GRDP",
    "Internet",
    "Gini",
]
IS_BENEFIT = np.array([True, True, True, True, True, True, True, False])
W_EXPERT = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10], dtype=float)

REGION_VI = {
    "Northern Midlands and Mountains": "Trung du miền núi phía Bắc",
    "Red River Delta": "Đồng bằng sông Hồng",
    "North Central and South Central Coast": "Bắc Trung Bộ + DH Trung Bộ",
    "Central Highlands": "Tây Nguyên",
    "Southeast": "Đông Nam Bộ",
    "Mekong Delta": "Đồng bằng sông Cửu Long",
}
REGION_SHORT = {
    "Northern Midlands and Mountains": "TDMNPB",
    "Red River Delta": "ĐBSH",
    "North Central and South Central Coast": "BTB+DHMT",
    "Central Highlands": "Tây Nguyên",
    "Southeast": "ĐNB",
    "Mekong Delta": "ĐBSCL",
}
REGION_COLORS = {
    "TDMNPB": "#1976D2",
    "ĐBSH": "#7B1FA2",
    "BTB+DHMT": "#FB8C00",
    "Tây Nguyên": "#1A6B3C",
    "ĐNB": "#E53935",
    "ĐBSCL": "#00ACC1",
}


# ── Dữ liệu fallback đúng theo đề ────────────────────────────────────────────
FALLBACK_ROWS = [
    [1, "Northern Midlands and Mountains", 57.0, 3.5, 38, 22, 21.5, 0.18, 72, 0.405],
    [2, "Red River Delta", 152.3, 20.0, 78, 68, 36.8, 0.85, 92, 0.358],
    [3, "North Central and South Central Coast", 87.5, 8.2, 55, 40, 27.5, 0.32, 84, 0.372],
    [4, "Central Highlands", 68.9, 0.8, 32, 18, 18.2, 0.15, 68, 0.412],
    [5, "Southeast", 158.9, 18.5, 82, 75, 42.5, 0.78, 94, 0.385],
    [6, "Mekong Delta", 80.5, 2.1, 48, 30, 16.8, 0.22, 78, 0.392],
]
FALLBACK_COLS = [
    "region_id", "region_name_en", "grdp_per_capita_million_VND", "fdi_registered_billion_USD",
    "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
    "internet_penetration_pct", "gini_coef",
]


@st.cache_data
def load_regions_safe() -> pd.DataFrame:
    """Đọc vietnam_regions_2024.csv; nếu không thấy file thì dùng bảng fallback từ đề."""
    try:
        from data_loader import load_regions
        df = load_regions()
        if set(CRITERIA + ["region_name_en"]).issubset(df.columns):
            return df.copy()
    except Exception:
        pass

    candidates = [
        "vietnam_regions_2024.csv",
        "data/vietnam_regions_2024.csv",
        "./vietnam_regions_2024.csv",
        "/mnt/data/vietnam_regions_2024(1).csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if set(CRITERIA + ["region_name_en"]).issubset(df.columns):
                return df.copy()

    return pd.DataFrame(FALLBACK_ROWS, columns=FALLBACK_COLS)


def vector_normalize(X: np.ndarray) -> np.ndarray:
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1e-12
    return X / denom


def topsis(X: np.ndarray, weights: np.ndarray, is_benefit: np.ndarray):
    """TOPSIS chuẩn: vector normalization → weighted matrix → ideal/anti-ideal → C*."""
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    R = vector_normalize(X)
    V = R * w
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    C_star = S_neg / np.maximum(S_star + S_neg, 1e-12)
    return C_star, S_star, S_neg, R, V, A_star, A_neg


def entropy_weights(X: np.ndarray):
    """
    Trọng số Entropy theo đúng gợi ý đề: P = X / sum(X).
    Lưu ý: hướng lợi ích/chi phí được xử lý ở bước TOPSIS, không xử lý trong bước trọng số.
    """
    X_pos = np.maximum(X.astype(float), 1e-12)
    P = X_pos / np.maximum(X_pos.sum(axis=0), 1e-12)
    k = 1.0 / np.log(X.shape[0])
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    d[d < 0] = 0
    if d.sum() == 0:
        w = np.ones(X.shape[1]) / X.shape[1]
    else:
        w = d / d.sum()
    return w, E, d


def rank_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores)


def rank_of(order: np.ndarray, i: int) -> int:
    return int(np.where(order == i)[0][0]) + 1


def ahp_from_priority_vector(priority: np.ndarray):
    """
    AHP đơn giản: xây ma trận so sánh cặp nhất quán từ vector ưu tiên chuyên gia.
    A_ij = w_i / w_j. Khi ma trận nhất quán tuyệt đối, CR = 0.
    """
    p = np.array(priority, dtype=float)
    p = p / p.sum()
    A = p[:, None] / p[None, :]

    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[idx].real)
    w = np.abs(eigvecs[:, idx].real)
    w = w / w.sum()

    n = len(p)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_TABLE.get(n, 1.49)
    CR = 0.0 if RI == 0 else CI / RI
    return A, w, lambda_max, CI, CR


def make_result_table(regions_short, C, S_star, S_neg):
    order = rank_order(C)
    df = pd.DataFrame({
        "Vùng": regions_short,
        "S* đến lý tưởng tốt": S_star,
        "S⁻ đến lý tưởng xấu": S_neg,
        "C* TOPSIS": C,
        "Hạng": [rank_of(order, i) for i in range(len(regions_short))],
    })
    return df.sort_values("Hạng").reset_index(drop=True)


def render():
    bai_header(
        so="6",
        ten="TOPSIS xếp hạng 6 vùng kinh tế theo mức độ ưu tiên đầu tư AI",
        mo_ta="Cài TOPSIS từ đầu, trọng số chuyên gia, Entropy, độ nhạy w_AI và AHP đơn giản",
        cap_do="TRUNG BÌNH",
        tools=["numpy", "pandas", "TOPSIS", "Entropy", "AHP"],
        thoi_luong="1.5 tuần",
    )

    df = load_regions_safe()
    X = df[CRITERIA].values.astype(float)
    regions_en = df["region_name_en"].tolist()
    reg_short = [REGION_SHORT.get(r, r) for r in regions_en]
    reg_vi = [REGION_VI.get(r, r) for r in regions_en]
    reg_colors = [REGION_COLORS.get(r, "#888") for r in reg_short]

    C_exp, S_exp, N_exp, R, V_exp, A_star, A_neg = topsis(X, W_EXPERT, IS_BENEFIT)
    order_exp = rank_order(C_exp)

    W_entropy, E_entropy, D_entropy = entropy_weights(X)
    C_ent, S_ent, N_ent, _, _, _, _ = topsis(X, W_entropy, IS_BENEFIT)
    order_ent = rank_order(C_ent)

    A_ahp, W_ahp, lambda_max, CI, CR = ahp_from_priority_vector(W_EXPERT)
    C_ahp, S_ahp, N_ahp, _, _, _, _ = topsis(X, W_ahp, IS_BENEFIT)
    order_ahp = rank_order(C_ahp)

    info_box(
        "TOPSIS đo mức gần phương án lý tưởng: <b>Cᵢ* = Sᵢ⁻ / (Sᵢ* + Sᵢ⁻)</b>. "
        "Các tiêu chí GRDP/người, FDI, Digital Index, AI Readiness, lao động đào tạo, R&D/GRDP, Internet là lợi ích; "
        "<b>Gini là tiêu chí chi phí</b> nên giá trị thấp tốt hơn.",
        bg="#E8F5E9", border=C1, icon="📐",
    )

    # ════════════════════════════════════════════════════════
    # CÂU 6.4.1 — TOPSIS chuyên gia
    # ════════════════════════════════════════════════════════
    section_title("Câu 6.4.1 — Cài đặt TOPSIS từ đầu bằng numpy", "🏆")

    tab_raw, tab_r, tab_v = st.tabs(["📋 Dữ liệu gốc", "🔢 Ma trận R chuẩn hóa", "⚖️ Ma trận V có trọng số"])
    with tab_raw:
        df_raw = pd.DataFrame(X, columns=CRITERIA_VI)
        df_raw.insert(0, "Vùng", reg_vi)
        st.dataframe(df_raw.round(4), use_container_width=True, hide_index=True)
    with tab_r:
        df_R = pd.DataFrame(R, columns=CRITERIA_VI)
        df_R.insert(0, "Vùng", reg_short)
        st.dataframe(df_R.round(4), use_container_width=True, hide_index=True)
        info_box("Chuẩn hóa vector: rᵢⱼ = xᵢⱼ / √Σᵢxᵢⱼ². Cách này giữ tỷ lệ tương đối giữa các vùng.",
                 bg="#F1F8F2", border=C2, icon="✅")
    with tab_v:
        df_V = pd.DataFrame(V_exp, columns=CRITERIA_VI)
        df_V.insert(0, "Vùng", reg_short)
        st.dataframe(df_V.round(5), use_container_width=True, hide_index=True)
        st.caption("V = R × w, với w chuyên gia = [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10].")

    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)

    medals = ["🥇", "🥈", "🥉"]
    mcolors = [C1, C2, C3]
    ctop = st.columns(3, gap="medium")
    for card, rank, medal, color in zip(ctop, range(3), medals, mcolors):
        idx = order_exp[rank]
        with card:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color},{color}cc);border-radius:14px;
                        padding:1.1rem 1.2rem;color:#fff;box-shadow:0 4px 16px {color}44;">
                <div style="font-size:1.8rem;margin-bottom:4px;">{medal}</div>
                <div style="font-size:1.05rem;font-weight:800;line-height:1.2;min-height:42px;">{reg_vi[idx]}</div>
                <div style="font-size:0.8rem;opacity:0.85;margin-top:8px;">C* chuyên gia</div>
                <div style="font-size:1.55rem;font-weight:900;">{C_exp[idx]:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    col_bar, col_tbl = st.columns([3, 2], gap="large")
    with col_bar:
        fig = go.Figure(go.Bar(
            x=[reg_short[i] for i in order_exp],
            y=[C_exp[i] for i in order_exp],
            marker_color=[reg_colors[i] for i in order_exp],
            text=[f"{C_exp[i]:.4f}" for i in order_exp],
            textposition="outside",
            marker_line_width=0,
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#8AA898")
        fig.update_layout(
            height=290, margin=dict(l=0, r=0, t=15, b=0), font=CF,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#F0F4F0"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col_tbl:
        st.dataframe(make_result_table(reg_short, C_exp, S_exp, N_exp).round(4), use_container_width=True, hide_index=True)

    info_box(
        f"Kết quả chuyên gia: <b>{reg_vi[order_exp[0]]}</b> đứng đầu (C*={C_exp[order_exp[0]]:.4f}), "
        f"sau đó là <b>{reg_vi[order_exp[1]]}</b> (C*={C_exp[order_exp[1]]:.4f}) và "
        f"<b>{reg_vi[order_exp[2]]}</b> (C*={C_exp[order_exp[2]]:.4f}). "
        "Lý do: hai vùng đầu có AI Readiness, Digital Index, FDI, GRDP/người và Internet đều vượt trội; "
        "vùng thứ ba có vị trí trung gian, không mạnh như hai cực nhưng cân bằng hơn các vùng còn lại.",
        bg="#E8F5E9", border=C3, icon="💡",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 6.4.2 — Entropy
    # ════════════════════════════════════════════════════════
    section_title("Câu 6.4.2 — Trọng số khách quan bằng Entropy", "🔢")

    col_w, col_cmp = st.columns([3, 2], gap="large")
    with col_w:
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(name="Chuyên gia", x=CRITERIA_VI, y=W_EXPERT, marker_color=CBLUE, marker_line_width=0))
        fig_w.add_trace(go.Bar(name="Entropy", x=CRITERIA_VI, y=W_entropy, marker_color=CRED, marker_line_width=0))
        fig_w.update_layout(
            height=310, margin=dict(l=0, r=0, t=15, b=0), barmode="group", font=CF,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.35),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F0"),
            xaxis=dict(tickangle=-30, showgrid=False),
        )
        st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar": False})
        df_w = pd.DataFrame({
            "Tiêu chí": CRITERIA_VI,
            "w chuyên gia": W_EXPERT,
            "Entropy E": E_entropy,
            "Độ phân biệt d=1-E": D_entropy,
            "w Entropy": W_entropy,
        })
        st.dataframe(df_w.round(4), use_container_width=True, hide_index=True)
    with col_cmp:
        df_cmp = pd.DataFrame({
            "Vùng": reg_short,
            "C* chuyên gia": C_exp,
            "Hạng CG": [rank_of(order_exp, i) for i in range(6)],
            "C* Entropy": C_ent,
            "Hạng Entropy": [rank_of(order_ent, i) for i in range(6)],
        })
        df_cmp["Δ hạng"] = df_cmp["Hạng CG"] - df_cmp["Hạng Entropy"]
        df_cmp = df_cmp.sort_values("Hạng Entropy")
        st.dataframe(df_cmp.round(4), use_container_width=True, hide_index=True)

    max_w_idx = int(np.argmax(W_entropy))
    info_box(
        f"Entropy gán trọng số cao nhất cho <b>{CRITERIA_VI[max_w_idx]}</b> (w={W_entropy[max_w_idx]:.4f}), "
        "vì tiêu chí này phân tán mạnh nhất giữa các vùng. Xếp hạng Entropy vẫn giữ cùng top-3 với trọng số chuyên gia, "
        "nhưng <b>Đồng bằng sông Hồng</b> vượt <b>Đông Nam Bộ</b> lên hạng 1 do FDI và R&D/GRDP được Entropy nhấn mạnh. "
        "Điều này cho thấy kết quả khá bền vững ở nhóm top-3, nhưng thứ tự #1 phụ thuộc triết lý trọng số.",
        bg="#FFF8E1", border=CORANGE, icon="🔍",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 6.4.3 — Độ nhạy w_AI
    # ════════════════════════════════════════════════════════
    section_title("Câu 6.4.3 — Phân tích độ nhạy w_AI từ 0.10 đến 0.40", "📈")

    ai_idx = 3
    w_ai_range = np.round(np.arange(0.10, 0.401, 0.05), 2)
    rank_data = {s: [] for s in reg_short}
    top3_rows = []
    for w_ai in w_ai_range:
        w = W_EXPERT.copy()
        w[ai_idx] = 0
        w = w / w.sum() * (1 - w_ai)
        w[ai_idx] = w_ai
        C_s, *_ = topsis(X, w, IS_BENEFIT)
        order_s = rank_order(C_s)
        top3_rows.append({
            "w_AI": w_ai,
            "Top 1": reg_short[order_s[0]],
            "Top 2": reg_short[order_s[1]],
            "Top 3": reg_short[order_s[2]],
            "C* #1": C_s[order_s[0]],
        })
        for i, s in enumerate(reg_short):
            rank_data[s].append(rank_of(order_s, i))

    fig_s = go.Figure()
    for s in reg_short:
        fig_s.add_trace(go.Scatter(
            x=w_ai_range, y=rank_data[s], mode="lines+markers", name=s,
            line=dict(color=REGION_COLORS.get(s, "#888"), width=2.5),
            marker=dict(size=7),
        ))
    fig_s.add_vline(x=0.20, line_dash="dash", line_color="#8AA898", annotation_text="mặc định 0.20")
    fig_s.update_layout(
        height=330, margin=dict(l=0, r=0, t=15, b=0), font=CF,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.27),
        yaxis=dict(autorange="reversed", dtick=1, title="Thứ hạng", showgrid=True, gridcolor="#F0F4F0"),
        xaxis=dict(title="w_AI", showgrid=False),
    )
    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(pd.DataFrame(top3_rows).round(4), use_container_width=True, hide_index=True)

    info_box(
        "Top-3 <b>ổn định tuyệt đối</b> trong dải w_AI = 0.10–0.40: luôn là Đông Nam Bộ, Đồng bằng sông Hồng, "
        "Bắc Trung Bộ + DH Trung Bộ. Khi w_AI tăng, Đông Nam Bộ càng củng cố vị trí #1 vì AI Readiness = 75/100 cao nhất; "
        "Tây Nguyên luôn cuối do đồng thời yếu ở AI, Digital, FDI và R&D.",
        bg="#E3F2FD", border=CBLUE, icon="✅",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 6.4.4 — AHP đơn giản
    # ════════════════════════════════════════════════════════
    section_title("Câu 6.4.4 — AHP đơn giản để so sánh với TOPSIS", "⚖️")

    col_ahp1, col_ahp2 = st.columns([3, 2], gap="large")
    with col_ahp1:
        st.markdown("**Ma trận so sánh cặp AHP cho 8 tiêu chí**")
        df_A = pd.DataFrame(A_ahp, columns=CRITERIA_VI, index=CRITERIA_VI)
        st.dataframe(df_A.round(2), use_container_width=True)
        st.caption("Ma trận được dựng từ tỷ lệ ưu tiên chuyên gia: aᵢⱼ = wᵢ / wⱼ. Vì vậy ma trận nhất quán tuyệt đối.")
    with col_ahp2:
        df_ahp_w = pd.DataFrame({
            "Tiêu chí": CRITERIA_VI,
            "w AHP": W_ahp,
            "w chuyên gia": W_EXPERT,
        })
        st.dataframe(df_ahp_w.round(4), use_container_width=True, hide_index=True)
        info_box(
            f"λmax = {lambda_max:.4f}; CI = {CI:.4f}; CR = {CR:.4f}. "
            "CR < 0.10 nên ma trận AHP đạt tính nhất quán.",
            bg="#F1F8F2", border=C2, icon="✅",
        )

    df_ahp_cmp = pd.DataFrame({
        "Vùng": reg_short,
        "C* TOPSIS chuyên gia": C_exp,
        "Hạng TOPSIS": [rank_of(order_exp, i) for i in range(6)],
        "C* TOPSIS-AHP": C_ahp,
        "Hạng AHP": [rank_of(order_ahp, i) for i in range(6)],
    }).sort_values("Hạng AHP")
    st.dataframe(df_ahp_cmp.round(4), use_container_width=True, hide_index=True)
    info_box(
        "Vì AHP ở đây được dùng để suy ra trọng số từ cùng hệ ưu tiên chuyên gia, kết quả TOPSIS-AHP gần như trùng với TOPSIS chuyên gia. "
        "Điểm quan trọng của AHP không phải đổi thứ hạng bằng mọi giá, mà là kiểm tra tính nhất quán của bộ trọng số trước khi đưa vào TOPSIS.",
        bg="#FFF8E1", border=CORANGE, icon="💡",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # THẢO LUẬN CHÍNH SÁCH
    # ════════════════════════════════════════════════════════
    section_title("Câu hỏi thảo luận chính sách", "💬")
    ai_corr = float(np.corrcoef(X[:, 3], X[:, 6])[0, 1])
    top3_vi = [reg_vi[order_exp[i]] for i in range(3)]

    q_data = [
        (
            "a)",
            "Vùng nào dẫn đầu theo TOPSIS với trọng số chuyên gia? Có nên triển khai trung tâm AI đầu tiên ở đó không?",
            f"<b>{top3_vi[0]}</b> dẫn đầu với C*={C_exp[order_exp[0]]:.4f}. Đây là ứng viên mạnh nhất về năng lực triển khai "
            "vì có AI Readiness, Digital Index, GRDP/người, Internet và lao động đào tạo rất cao. Tuy nhiên, quyết định đặt trung tâm AI quốc gia "
            "không nên chỉ dựa vào điểm kỹ thuật; cần thêm tiêu chí an ninh dữ liệu, quỹ đất, hạ tầng điện, khả năng disaster recovery và cân bằng Bắc–Nam.",
        ),
        (
            "b)",
            "Khi dùng trọng số Entropy, vùng nào thay đổi xếp hạng lớn nhất? Vì sao?",
            "Không có vùng nào đảo hạng mạnh; mức thay đổi lớn nhất chỉ là 1 bậc. Về chính sách, thay đổi đáng chú ý nhất là "
            "<b>Đồng bằng sông Hồng</b> vượt <b>Đông Nam Bộ</b> lên #1. Lý do là Entropy gán trọng số rất cao cho FDI và R&D/GRDP, "
            "hai tiêu chí mà ĐBSH có lợi thế rõ. Điều này cho thấy Entropy phản ánh độ phân tán dữ liệu, còn trọng số chuyên gia phản ánh mục tiêu chiến lược.",
        ),
        (
            "c)",
            f"AI Readiness và Internet penetration tương quan cao (r={ai_corr:.3f}) ảnh hưởng thế nào?",
            f"Hệ số tương quan r={ai_corr:.3f} là rất cao, nên nếu giữ cả hai tiêu chí với trọng số riêng thì mô hình có nguy cơ <b>đếm hai lần</b> cùng một năng lực số. "
            "Kết quả sẽ thiên về các vùng đã phát triển hạ tầng số, làm giảm cơ hội của vùng đang lên. Cách xử lý: kiểm tra ma trận tương quan/VIF; "
            "gộp bằng PCA; giảm trọng số Internet; hoặc dùng phiên bản TOPSIS điều chỉnh tương quan tiêu chí.",
        ),
        (
            "d)",
            "Nếu Việt Nam xây 3 trung tâm AI lớn, nên chọn 3 vùng nào? Có cần điều chỉnh địa - chính trị không?",
            f"Theo TOPSIS thuần túy, 3 vùng là <b>{', '.join(top3_vi)}</b>. Nhưng nên điều chỉnh theo địa - chính trị: "
            "không tập trung cả 3 trung tâm ở một cực tăng trưởng; bảo đảm một trung tâm ở phía Bắc, một ở phía Nam, một ở miền Trung; xét thêm an ninh quốc phòng, "
            "điện năng, cáp quang biển, rủi ro thiên tai và khả năng kết nối đại học - doanh nghiệp.",
        ),
    ]
    for code, q, ans in q_data:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()
