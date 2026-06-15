"""pages/bai3.py — Bài 3: Chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam

Bản làm lại:
- Đọc vietnam_sectors_2024.csv thay vì hard-code toàn bộ dữ liệu.
- Tính Năng suất từ GDP share và lao động nếu file CSV không có cột productivity.
- Chuẩn hóa min-max đúng 7 tiêu chí.
- Xử lý Risk nhất quán: Risk là chỉ số xấu nên hoặc (i) chuẩn hóa risk rồi TRỪ,
  hoặc (ii) đảo thành điểm an toàn rồi CỘNG. File này dùng cách (ii).
- Phân tích độ nhạy a6 với tổng trọng số chuẩn hóa = 1.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from utils import bai_header, end_padding, info_box, section_title
except Exception:  # fallback để file vẫn chạy độc lập khi chưa có utils.py
    def bai_header(so, ten, mo_ta, cap_do, tools, thoi_luong):
        st.title(f"Bài {so}. {ten}")
        st.caption(mo_ta)
    def end_padding():
        st.markdown("<br><br>", unsafe_allow_html=True)
    def info_box(text, bg="#F5F5F5", border="#CCCCCC", icon="ℹ️"):
        st.markdown(
            f"""
            <div style="background:{bg};border-left:5px solid {border};border-radius:10px;
                        padding:0.9rem 1rem;margin:0.7rem 0;">
                <span style="font-size:1.05rem;">{icon}</span>&nbsp;{text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    def section_title(text, icon=""):
        st.subheader(f"{icon} {text}")


# ── Style ────────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"; C2 = "#2E8B57"; C3 = "#4CAF72"
CBLUE = "#1976D2"; CRED = "#E53935"; CORANGE = "#E65100"; CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

GDP_2024_TRILLION_VND = 11511.9  # theo đề, dùng để suy ra năng suất từ GDP share / lao động

SECTOR_NAME_VI = {
    "Agriculture-Forestry-Fishery": "Nông-Lâm-Thủy sản",
    "Manufacturing": "CN chế biến chế tạo",
    "Construction": "Xây dựng",
    "Mining": "Khai khoáng",
    "Wholesale-Retail": "Bán buôn-bán lẻ",
    "Finance-Banking-Insurance": "Tài chính-Ngân hàng",
    "Logistics-Transport-Warehousing": "Logistics-Vận tải",
    "Information-Communication-IT": "CNTT-Truyền thông",
    "Education-Training": "Giáo dục-Đào tạo",
    "Healthcare": "Y tế",
}

CRITERIA = [
    "growth_norm",
    "productivity_norm",
    "spillover_norm",
    "export_norm",
    "employment_norm",
    "ai_readiness_norm",
    "risk_safe_norm",
]
CRITERIA_LABELS = [
    "Tăng trưởng",
    "Năng suất",
    "Lan tỏa",
    "Xuất khẩu",
    "Việc làm",
    "AI Readiness",
    "Risk đảo / An toàn",
]


def _find_csv() -> Path:
    """Tìm file dữ liệu trong các vị trí thường gặp khi chạy Streamlit/Colab/local."""
    candidates = [
        Path("vietnam_sectors_2024.csv"),
        Path("data/vietnam_sectors_2024.csv"),
        Path("/mnt/data/vietnam_sectors_2024.csv"),
        Path("/mnt/data/vietnam_sectors_2024(1).csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Không tìm thấy vietnam_sectors_2024.csv. Hãy đặt file CSV cùng thư mục chạy app "
        "hoặc trong thư mục data/."
    )


def _norm_good(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / rng


def _norm_bad_to_safe(s: pd.Series) -> pd.Series:
    """Risk là chỉ số xấu: risk thấp phải nhận điểm cao."""
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s.max() - s) / rng


@st.cache_data
def _load_and_prepare() -> pd.DataFrame:
    path = _find_csv()
    df = pd.read_csv(path)

    # Tên ngành tiếng Việt
    if "sector_name_vi" not in df.columns:
        if "sector_name_en" in df.columns:
            df["sector_name_vi"] = df["sector_name_en"].map(SECTOR_NAME_VI).fillna(df["sector_name_en"])
        else:
            df["sector_name_vi"] = [f"Ngành {i+1}" for i in range(len(df))]

    # Năng suất trong đề = GDP ngành / lao động.
    # File CSV thường chỉ có gdp_share_2024_pct và labor_million, nên ta suy ra productivity.
    if "productivity_million_VND_per_worker" not in df.columns:
        required = {"gdp_share_2024_pct", "labor_million"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Thiếu cột để tính năng suất: {missing}")
        df["productivity_million_VND_per_worker"] = (
            df["gdp_share_2024_pct"] / 100 * GDP_2024_TRILLION_VND / df["labor_million"]
        )

    # Chuẩn hóa 6 tiêu chí tốt
    df["growth_norm"] = _norm_good(df["growth_rate_2024_pct"])
    df["productivity_norm"] = _norm_good(df["productivity_million_VND_per_worker"])
    df["spillover_norm"] = _norm_good(df["spillover_coef_0_1"])
    df["export_norm"] = _norm_good(df["export_billion_USD"])
    df["employment_norm"] = _norm_good(df["labor_million"])
    df["ai_readiness_norm"] = _norm_good(df["ai_readiness_0_100"])

    # Risk đảo thành điểm an toàn: risk thấp → điểm cao.
    df["risk_safe_norm"] = _norm_bad_to_safe(df["automation_risk_pct"])
    # Cột này chỉ để kiểm tra tương đương đại số nếu muốn dùng công thức trừ Risk xấu.
    df["risk_bad_norm"] = _norm_good(df["automation_risk_pct"])

    return df


def _priority(df: pd.DataFrame, weights_good: np.ndarray, weight_risk_safe: float) -> np.ndarray:
    """Tính Priority theo cách nhất quán:

    Priority = sum(w_good * good_norm) + w_risk * risk_safe_norm

    Cách này tương đương về xếp hạng với: sum(w_good * good_norm) - w_risk * risk_bad_norm,
    chỉ khác nhau một hằng số +w_risk cho mọi ngành. Tuyệt đối không vừa đảo Risk vừa trừ Risk đảo.
    """
    X_good = df[[
        "growth_norm", "productivity_norm", "spillover_norm", "export_norm",
        "employment_norm", "ai_readiness_norm",
    ]].to_numpy(dtype=float)
    risk_safe = df["risk_safe_norm"].to_numpy(dtype=float)
    return X_good @ weights_good + weight_risk_safe * risk_safe


def _rank_table(df: pd.DataFrame, score: np.ndarray, col_name: str = "Priority") -> pd.DataFrame:
    out = df[["sector_name_vi"]].copy()
    out[col_name] = score
    out = out.sort_values(col_name, ascending=False).reset_index(drop=True)
    out.insert(0, "Hạng", np.arange(1, len(out) + 1))
    return out


def _make_bar(y_labels, x_values, color=C1, height=340, x_title="Priority"):
    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_labels,
        orientation="h",
        marker_color=color,
        marker_line_width=0,
        text=[f"{v:.4f}" for v in x_values],
        textposition="outside",
        hovertemplate="%{y}<br>Priority: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=55, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CF,
        showlegend=False,
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=11)),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#F0F4F0", zeroline=True),
    )
    return fig


def render():
    bai_header(
        so="3",
        ten="Tính chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam",
        mo_ta="Đọc dữ liệu, chuẩn hóa min-max, xử lý Risk đúng dấu, xếp hạng ngành và phân tích độ nhạy trọng số AI Readiness",
        cap_do="DỄ",
        tools=["pandas", "numpy", "plotly"],
        thoi_luong="1 tuần",
    )

    df = _load_and_prepare()

    # Trọng số mặc định theo đề. Tổng = 1.10 vì đề cho a1...a7 như vậy;
    # ta giữ nguyên để bám sát đề ở câu 3.4.2.
    w_default = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20], dtype=float)
    w_risk_default = 0.15
    priority_default = _priority(df, w_default, w_risk_default)
    ranking_default = _rank_table(df, priority_default)

    info_box(
        "Công thức triển khai nhất quán: <b>Priorityᵢ = Σ w·tiêu_chí_tốt + a₇·Risk_safe</b>, "
        "trong đó <b>Risk_safe = (max Risk − Riskᵢ)/(max Risk − min Risk)</b>.<br>"
        "Cách này tương đương về xếp hạng với việc chuẩn hóa Risk theo chiều xấu rồi trừ Risk. "
        "Lỗi cần tránh: <b>đảo Risk rồi lại trừ Risk đảo</b>, vì như vậy sẽ phạt ngành rủi ro thấp.",
        bg="#E8F5E9", border=C1, icon="📐"
    )

    # ════════════════════════════════════════════════════════
    # CÂU 3.4.1
    # ════════════════════════════════════════════════════════
    section_title("Câu 3.4.1 — Đọc dữ liệu và chuẩn hóa min-max 7 tiêu chí", "📊")

    raw_cols = [
        "sector_name_vi", "growth_rate_2024_pct", "productivity_million_VND_per_worker",
        "spillover_coef_0_1", "export_billion_USD", "labor_million",
        "ai_readiness_0_100", "automation_risk_pct",
    ]
    df_raw = df[raw_cols].rename(columns={
        "sector_name_vi": "Ngành",
        "growth_rate_2024_pct": "Tăng trưởng (%)",
        "productivity_million_VND_per_worker": "Năng suất (tr.VND/LĐ)",
        "spillover_coef_0_1": "Lan tỏa",
        "export_billion_USD": "Xuất khẩu (tỷ USD)",
        "labor_million": "Việc làm (triệu LĐ)",
        "ai_readiness_0_100": "AI Readiness",
        "automation_risk_pct": "Rủi ro TĐH (%)",
    }).copy()
    df_raw["Năng suất (tr.VND/LĐ)"] = df_raw["Năng suất (tr.VND/LĐ)"].round(1)

    df_norm = df[["sector_name_vi"] + CRITERIA].rename(columns={"sector_name_vi": "Ngành"}).copy()
    df_norm.columns = ["Ngành"] + CRITERIA_LABELS
    for c in CRITERIA_LABELS:
        df_norm[c] = df_norm[c].round(4)

    tab_raw, tab_norm, tab_check = st.tabs(["📋 Dữ liệu gốc", "🔢 Ma trận chuẩn hóa", "✅ Kiểm tra dấu Risk"])
    with tab_raw:
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
        info_box(
            "Nếu CSV không có cột năng suất, file tự tính: "
            "<b>Năng suất = GDP share × 11.511,9 / lao động</b>. "
            "Cách tính này khớp bảng đề: ví dụ Khai khoáng ≈ 1.290,5 tr.VND/LĐ.",
            bg="#E3F2FD", border=CBLUE, icon="ℹ️"
        )
    with tab_norm:
        st.dataframe(df_norm, use_container_width=True, hide_index=True)
    with tab_check:
        risk_check = df[["sector_name_vi", "automation_risk_pct", "risk_bad_norm", "risk_safe_norm"]].copy()
        risk_check.columns = ["Ngành", "Risk gốc (%)", "Risk_norm_xấu", "Risk_safe_đảo"]
        risk_check["Quan hệ"] = (risk_check["Risk_norm_xấu"] + risk_check["Risk_safe_đảo"]).round(6)
        st.dataframe(risk_check.round(4), use_container_width=True, hide_index=True)
        info_box(
            "Vì <b>Risk_norm_xấu + Risk_safe_đảo = 1</b>, nên "
            "ΣwX − a₇·Risk_norm_xấu và ΣwX + a₇·Risk_safe_đảo chỉ khác nhau một hằng số a₇. "
            "Hai cách cho cùng xếp hạng. Nhưng <b>ΣwX − a₇·Risk_safe_đảo</b> là sai logic.",
            bg="#FFF8E1", border=CORANGE, icon="⚠️"
        )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 3.4.2
    # ════════════════════════════════════════════════════════
    section_title("Câu 3.4.2 — Tính Priorityᵢ và xếp hạng theo trọng số mặc định", "🏆")

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    total_w = w_default.sum() + w_risk_default
    top_idx = ranking_default.index[:3]
    top_rows = ranking_default.iloc[:3]
    kpis = [
        (c1, "Tổng trọng số đề cho", f"{total_w:.2f}", "giữ nguyên theo đề", CBLUE),
        (c2, "Top 1", top_rows.iloc[0]["sector_name_vi"], f"{top_rows.iloc[0]['Priority']:.4f}", C1),
        (c3, "Top 2", top_rows.iloc[1]["sector_name_vi"], f"{top_rows.iloc[1]['Priority']:.4f}", C2),
        (c4, "Top 3", top_rows.iloc[2]["sector_name_vi"], f"{top_rows.iloc[2]['Priority']:.4f}", C3),
    ]
    for col, label, value, sub, color in kpis:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:1rem 1.1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);height:105px;">
                <div style="font-size:0.75rem;font-weight:800;color:#6B8A7A;text-transform:uppercase;margin-bottom:6px;">{label}</div>
                <div style="font-size:1.05rem;font-weight:900;color:{color};line-height:1.15;">{value}</div>
                <div style="font-size:0.78rem;color:#8AA898;margin-top:5px;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

    col_rank, col_decomp = st.columns([3, 2], gap="large")
    with col_rank:
        table_rank = ranking_default.copy()
        table_rank["Priority"] = table_rank["Priority"].round(4)
        table_rank = table_rank.rename(columns={"sector_name_vi": "Ngành"})
        st.dataframe(table_rank, use_container_width=True, hide_index=True)

        fig_rank = _make_bar(
            y_labels=table_rank["Ngành"].tolist(),
            x_values=table_rank["Priority"].tolist(),
            color=[C1 if i < 3 else C3 for i in range(len(table_rank))],
            height=360,
        )
        st.plotly_chart(fig_rank, use_container_width=True, config={"displayModeBar": False})

    with col_decomp:
        top_sector = ranking_default.iloc[0]["sector_name_vi"]
        top_row = df[df["sector_name_vi"] == top_sector].iloc[0]
        contrib = pd.DataFrame({
            "Tiêu chí": CRITERIA_LABELS,
            "Điểm chuẩn hóa": top_row[CRITERIA].to_numpy(dtype=float),
            "Trọng số": list(w_default) + [w_risk_default],
        })
        contrib["Đóng góp"] = contrib["Điểm chuẩn hóa"] * contrib["Trọng số"]
        st.markdown(f"**Phân rã điểm của ngành dẫn đầu: {top_sector}**")
        st.dataframe(contrib.round(4), use_container_width=True, hide_index=True)
        info_box(
            "CNTT-Truyền thông đứng đầu vì đạt điểm rất cao ở <b>AI Readiness</b>, <b>lan tỏa</b> "
            "và <b>xuất khẩu</b>. CN chế biến chế tạo vẫn rất mạnh nhờ quy mô xuất khẩu và việc làm, "
            "nhưng bị kéo xuống bởi rủi ro tự động hóa cao hơn.",
            bg="#E8F5E9", border=C3, icon="💡"
        )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 3.4.3
    # ════════════════════════════════════════════════════════
    section_title("Câu 3.4.3 — Phân tích độ nhạy khi thay đổi trọng số AI Readiness a₆", "🔥")

    # Chuẩn hóa lại tổng = 1 theo yêu cầu câu 3.4.3:
    # giữ a7 = 0.15, thay a6, các trọng số a1-a5 giữ tỷ lệ gốc rồi scale phần còn lại.
    a6_values = np.round(np.arange(0.05, 0.401, 0.05), 2)
    base_a1_a5 = np.array([0.15, 0.15, 0.20, 0.15, 0.10], dtype=float)
    risk_fixed = 0.15
    sens_matrix = np.zeros((len(df), len(a6_values)))
    top3_by_a6 = []

    for j, a6 in enumerate(a6_values):
        remain = 1.0 - risk_fixed - a6
        if remain < 0:
            raise ValueError("a6 quá lớn khiến tổng trọng số vượt 1")
        w_a1_a5 = base_a1_a5 / base_a1_a5.sum() * remain
        w = np.r_[w_a1_a5, a6]
        p = _priority(df, w, risk_fixed)
        sens_matrix[:, j] = p
        order = np.argsort(-p)[:3]
        top3_by_a6.append([df.iloc[i]["sector_name_vi"] for i in order])

    avg_sens = sens_matrix.mean(axis=1)
    sort_idx = np.argsort(-avg_sens)

    fig_hm = go.Figure(go.Heatmap(
        z=sens_matrix[sort_idx],
        x=[f"{v:.2f}" for v in a6_values],
        y=df.iloc[sort_idx]["sector_name_vi"],
        colorscale=[[0, "#FFEBEE"], [0.35, "#FFF8E1"], [0.65, "#A5D6A7"], [1, "#1A6B3C"]],
        text=np.round(sens_matrix[sort_idx], 3),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Priority", thickness=12, len=0.82),
        hovertemplate="%{y}<br>a₆=%{x}<br>Priority: %{z:.4f}<extra></extra>",
    ))
    fig_hm.update_layout(
        height=390,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CF,
        xaxis=dict(title="a₆ — trọng số AI Readiness"),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

    df_top3 = pd.DataFrame(top3_by_a6, columns=["Top 1", "Top 2", "Top 3"])
    df_top3.insert(0, "a₆", [f"{v:.2f}" for v in a6_values])
    st.dataframe(df_top3, use_container_width=True, hide_index=True)

    info_box(
        "Kết quả nhạy nhưng không đảo lộn mạnh: <b>Top-3 luôn là CNTT-Truyền thông, CN chế biến chế tạo "
        "và Tài chính-Ngân hàng</b>. Khi a₆ tăng, CNTT càng cách biệt vì AI Readiness = 88 cao nhất. "
        "Điều này cho thấy kết luận ưu tiên nhóm ngành công nghệ-sản xuất-tài chính là khá robust.",
        bg="#FFF8E1", border=CORANGE, icon="🔍"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 3.4.4
    # ════════════════════════════════════════════════════════
    section_title("Câu 3.4.4 — So sánh hai bộ trọng số chính sách", "⚖️")

    # Hai bộ trọng số tự xây dựng, tổng = 1, dùng Risk_safe như một tiêu chí tốt.
    # Thứ tự: Growth, Productivity, Spillover, Export, Employment, AI, Risk_safe
    weights_growth = np.array([0.25, 0.25, 0.10, 0.20, 0.05, 0.10], dtype=float)
    risk_growth = 0.05
    weights_inclusive = np.array([0.10, 0.05, 0.25, 0.05, 0.25, 0.10], dtype=float)
    risk_inclusive = 0.20

    p_growth = _priority(df, weights_growth, risk_growth)
    p_inclusive = _priority(df, weights_inclusive, risk_inclusive)
    rank_growth = _rank_table(df, p_growth, "Priority tăng trưởng")
    rank_inclusive = _rank_table(df, p_inclusive, "Priority bao trùm")

    col_g, col_i = st.columns(2, gap="large")
    with col_g:
        st.markdown(f"<h4 style='color:{CBLUE};'>📈 Định hướng tăng trưởng</h4>", unsafe_allow_html=True)
        st.caption("Ưu tiên tăng trưởng, năng suất, xuất khẩu; vẫn giữ một phần AI và rủi ro.")
        st.dataframe(rank_growth.head(5).round(4).rename(columns={"sector_name_vi": "Ngành"}),
                     use_container_width=True, hide_index=True)
        fig_g = _make_bar(rank_growth.head(5)["sector_name_vi"].tolist(),
                          rank_growth.head(5)["Priority tăng trưởng"].tolist(),
                          color=CBLUE, height=250)
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

    with col_i:
        st.markdown(f"<h4 style='color:{CPURPLE};'>🤝 Định hướng bao trùm</h4>", unsafe_allow_html=True)
        st.caption("Ưu tiên việc làm, lan tỏa và giảm rủi ro tự động hóa.")
        st.dataframe(rank_inclusive.head(5).round(4).rename(columns={"sector_name_vi": "Ngành"}),
                     use_container_width=True, hide_index=True)
        fig_i = _make_bar(rank_inclusive.head(5)["sector_name_vi"].tolist(),
                          rank_inclusive.head(5)["Priority bao trùm"].tolist(),
                          color=CPURPLE, height=250)
        st.plotly_chart(fig_i, use_container_width=True, config={"displayModeBar": False})

    info_box(
        "Hai bộ trọng số cho thông điệp chính sách khác nhau. <b>Định hướng tăng trưởng</b> chọn CNTT, "
        "CN chế biến chế tạo, Tài chính vì năng suất/xuất khẩu/sẵn sàng AI cao. <b>Định hướng bao trùm</b> "
        "đưa Nông-Lâm-Thủy sản vào Top-3 vì việc làm lớn và rủi ro tự động hóa thấp. "
        "Vì vậy trọng số không phải chi tiết kỹ thuật trung lập; nó là lựa chọn chính sách.",
        bg="#E3F2FD", border=CBLUE, icon="💡"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # THẢO LUẬN CHÍNH SÁCH
    # ════════════════════════════════════════════════════════
    section_title("Câu hỏi thảo luận chính sách", "💬")

    q_data = [
        (
            "a)",
            "Ba ngành nào nên ưu tiên đẩy mạnh chuyển đổi số và AI trước?",
            "Theo cách xử lý Risk đúng dấu, Top-3 mặc định là <b>CNTT-Truyền thông, CN chế biến chế tạo, "
            "Tài chính-Ngân hàng</b>. Kết quả này hợp lý vì đây là các ngành có năng lực hấp thụ AI cao, "
            "lan tỏa lớn và có khả năng kéo các ngành khác thông qua hạ tầng số, chuỗi cung ứng, thanh toán, "
            "dữ liệu và tự động hóa quản trị. Về logic chính sách, kết quả này phù hợp tinh thần Nghị quyết 57 "
            "ở điểm ưu tiên khoa học-công nghệ, đổi mới sáng tạo và chuyển đổi số như động lực tăng trưởng mới.",
        ),
        (
            "b)",
            "Tại sao Khai khoáng có năng suất rất cao nhưng vẫn không nằm trong nhóm ưu tiên?",
            "Vì đây là bài toán đa tiêu chí, không phải xếp hạng theo một biến. Khai khoáng có năng suất cao "
            "chủ yếu do thâm dụng vốn/tài nguyên và lao động ít, nhưng bị trừ mạnh bởi tăng trưởng âm, lan tỏa thấp, "
            "việc làm rất nhỏ và rủi ro tự động hóa cao. Nói cách khác, năng suất cao không đồng nghĩa với khả năng "
            "tạo hiệu ứng chuyển đổi số rộng cho nền kinh tế.",
        ),
        (
            "c)",
            "Bộ trọng số nên do ai quyết định?",
            "Nên dùng governance đa tầng. Chuyên gia kỹ thuật thiết kế chỉ tiêu, kiểm tra dữ liệu và phân tích độ nhạy; "
            "hội đồng chính sách quyết định trọng số trên cơ sở mục tiêu phát triển; đối thoại công khai giúp tăng tính "
            "chính danh và giảm thiên lệch lợi ích nhóm. Phân tích độ nhạy ở câu 3.4.3 là bước bắt buộc vì nó cho thấy "
            "kết luận có ổn định hay bị thao túng bởi một lựa chọn trọng số duy nhất.",
        ),
    ]
    for code, question, answer in q_data:
        with st.expander(f"{code} {question}", expanded=False):
            info_box(answer, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()


if __name__ == "__main__":
    render()
