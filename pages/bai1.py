"""pages/bai1.py — Bài 1: Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa

Bản làm lại:
- Đọc GDP và tỷ trọng kinh tế số từ file CSV khi có thể.
- Bổ sung K, L, AI, H theo bảng dữ liệu của đề bài.
- Tính đúng A_t, A trung bình, Y_hat, MAPE, phân rã tăng trưởng và dự báo GDP 2030.
- Sửa lỗi KPI bị lặp, nhận xét sai năm có sai số lớn nhất, pie chart dùng abs(), và đường dự báo 2025–2030 bị nhảy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from data_loader import load_macro
except Exception:  # pragma: no cover
    load_macro = None

try:
    from utils import bai_header, end_padding, info_box, section_title
except Exception:  # pragma: no cover
    def bai_header(so, ten, mo_ta, cap_do, tools, thoi_luong):
        st.title(f"Bài {so}. {ten}")
        st.caption(f"{mo_ta} · Cấp độ: {cap_do} · Công cụ: {', '.join(tools)} · Thời lượng: {thoi_luong}")

    def section_title(title, icon=""):
        st.subheader(f"{icon} {title}")

    def info_box(text, bg="#F7F9FA", border="#CBD5E1", icon="ℹ️"):
        st.markdown(
            f"""
            <div style="background:{bg};border-left:5px solid {border};padding:0.9rem 1rem;
                        border-radius:10px;margin:0.7rem 0;color:#1A2B1F;">
                <b>{icon}</b>&nbsp;{text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def end_padding():
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)


# ── Style ────────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"
C2 = "#2E8B57"
C3 = "#4CAF72"
ORANGE = "#E65100"
BLUE = "#1976D2"
PURPLE = "#7B1FA2"
RED = "#E53935"
CF = dict(family="Montserrat, Arial, sans-serif", size=12)

# ── Hệ số Cobb-Douglas theo đề bài ───────────────────────────────────────────
ALPHA = 0.33   # K
BETA = 0.42    # L
GAMMA = 0.10   # D
DELTA = 0.08   # AI
THETA = 0.07   # H
COEFS = {
    "Vốn (K)": ALPHA,
    "Lao động (L)": BETA,
    "Số hóa (D)": GAMMA,
    "AI": DELTA,
    "Nhân lực số (H)": THETA,
}

# Các biến chưa có trong file macro CSV nên khai báo theo bảng 1.3 của đề.
# Đơn vị: K = nghìn tỷ VND; L = triệu lao động; AI = nghìn DN số; H = % lao động qua đào tạo.
INPUTS_FROM_DE = pd.DataFrame(
    {
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "capital_accum_trillion_VND": [16500, 17800, 19600, 21300, 23500, 25900],
        "labor_million": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "ai_enterprises_thousand": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "trained_labor_pct": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    }
)

FALLBACK_MACRO = pd.DataFrame(
    {
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "digital_economy_share_GDP_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
    }
)


def _read_macro_csv() -> pd.DataFrame:
    """Đọc dữ liệu macro từ app/CSV; nếu không có thì dùng dữ liệu fallback theo đề."""
    if load_macro is not None:
        try:
            df = load_macro()
            if isinstance(df, pd.DataFrame) and {"year", "GDP_trillion_VND"}.issubset(df.columns):
                return df.copy()
        except Exception:
            pass

    candidates = [
        Path("vietnam_macro_2020_2025.csv"),
        Path("data/vietnam_macro_2020_2025.csv"),
        Path("datasets/vietnam_macro_2020_2025.csv"),
        Path("/mnt/data/vietnam_macro_2020_2025.csv"),
        Path("/mnt/data/vietnam_macro_2020_2025(1).csv"),
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    return FALLBACK_MACRO.copy()


def load_bai1_data() -> pd.DataFrame:
    """Chuẩn hóa dữ liệu đầu vào cho bài 1."""
    macro = _read_macro_csv()
    macro = macro.rename(columns={"Year": "year", "Năm": "year"})

    needed = ["year", "GDP_trillion_VND", "digital_economy_share_GDP_pct"]
    missing = [col for col in needed if col not in macro.columns]
    if missing:
        # Nếu file CSV thiếu cột cần thiết, dùng bảng fallback để tránh app lỗi.
        macro = FALLBACK_MACRO.copy()

    df = macro[needed].copy()
    df = df.merge(INPUTS_FROM_DE, on="year", how="left")
    df = df.sort_values("year").reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != "year"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if df[numeric_cols].isna().any().any():
        raise ValueError("Dữ liệu Bài 1 bị thiếu hoặc không chuyển được sang số. Hãy kiểm tra file CSV.")

    return df


def compute_bai1() -> Dict[str, object]:
    """Tính toàn bộ kết quả Bài 1."""
    df = load_bai1_data()

    years = df["year"].to_numpy(dtype=int)
    Y = df["GDP_trillion_VND"].to_numpy(dtype=float)
    K = df["capital_accum_trillion_VND"].to_numpy(dtype=float)
    L = df["labor_million"].to_numpy(dtype=float)
    D = df["digital_economy_share_GDP_pct"].to_numpy(dtype=float)
    AI = df["ai_enterprises_thousand"].to_numpy(dtype=float)
    H = df["trained_labor_pct"].to_numpy(dtype=float)

    core = (K ** ALPHA) * (L ** BETA) * (D ** GAMMA) * (AI ** DELTA) * (H ** THETA)
    A = Y / core
    A_bar = float(A.mean())
    Yhat = A_bar * core
    ape = np.abs(Y - Yhat) / Y * 100
    mape = float(ape.mean())
    max_error_idx = int(np.argmax(ape))

    # Phân rã tăng trưởng bình quân 2020–2025 theo log.
    n_years = int(years[-1] - years[0])
    dlnY = float(np.log(Y[-1] / Y[0]) / n_years)
    contrib = {
        "Vốn (K)": ALPHA * float(np.log(K[-1] / K[0]) / n_years),
        "Lao động (L)": BETA * float(np.log(L[-1] / L[0]) / n_years),
        "Số hóa (D)": GAMMA * float(np.log(D[-1] / D[0]) / n_years),
        "AI": DELTA * float(np.log(AI[-1] / AI[0]) / n_years),
        "Nhân lực số (H)": THETA * float(np.log(H[-1] / H[0]) / n_years),
    }
    contrib["TFP (A)"] = dlnY - sum(contrib.values())

    contrib_df = pd.DataFrame(
        {
            "Yếu tố": list(contrib.keys()),
            "Đóng góp (điểm %/năm)": [v * 100 for v in contrib.values()],
            "Tỷ trọng trong tăng trưởng (%)": [(v / dlnY) * 100 for v in contrib.values()],
        }
    )

    # Dự báo 2030: D = 30%, AI = 100 nghìn DN, H = 35%, K và L +6%/năm, A +1,2%/năm.
    forecast_years = np.arange(years[-1], 2031)
    t = np.arange(0, len(forecast_years))
    t_end = int(2030 - years[-1])

    K_fore = K[-1] * (1.06 ** t)
    L_fore = L[-1] * (1.06 ** t)
    A_fore = A[-1] * (1.012 ** t)

    # Nội suy D, AI, H để đường dự báo nối mượt từ 2025 đến 2030.
    D_fore = np.linspace(D[-1], 30.0, len(forecast_years))
    AI_fore = np.linspace(AI[-1], 100.0, len(forecast_years))
    H_fore = np.linspace(H[-1], 35.0, len(forecast_years))

    Y_fore = A_fore * (K_fore ** ALPHA) * (L_fore ** BETA) * (D_fore ** GAMMA) * (AI_fore ** DELTA) * (H_fore ** THETA)
    Y2030 = float(Y_fore[-1])
    cagr_2025_2030 = float(((Y2030 / Y[-1]) ** (1 / t_end) - 1) * 100)

    df_tfp = pd.DataFrame(
        {
            "Năm": years,
            "Y thực tế (nghìn tỷ VND)": Y,
            "A_t (TFP)": A,
            "Tăng/giảm A_t (%)": np.r_[np.nan, (A[1:] / A[:-1] - 1) * 100],
        }
    )

    df_fit = pd.DataFrame(
        {
            "Năm": years,
            "Y thực tế": Y,
            "Y dự báo": Yhat,
            "Sai số tuyệt đối": np.abs(Y - Yhat),
            "APE (%)": ape,
        }
    )

    df_forecast = pd.DataFrame(
        {
            "Năm": forecast_years,
            "K": K_fore,
            "L": L_fore,
            "D": D_fore,
            "AI": AI_fore,
            "H": H_fore,
            "A": A_fore,
            "GDP dự báo": Y_fore,
        }
    )

    return {
        "data": df,
        "years": years,
        "Y": Y,
        "K": K,
        "L": L,
        "D": D,
        "AI": AI,
        "H": H,
        "A": A,
        "A_bar": A_bar,
        "Yhat": Yhat,
        "APE": ape,
        "MAPE": mape,
        "max_error_idx": max_error_idx,
        "dlnY": dlnY,
        "contrib": contrib,
        "contrib_df": contrib_df,
        "forecast_years": forecast_years,
        "Y_fore": Y_fore,
        "Y2030": Y2030,
        "cagr_2025_2030": cagr_2025_2030,
        "df_tfp": df_tfp,
        "df_fit": df_fit,
        "df_forecast": df_forecast,
    }


def _metric_card(label: str, value: str, sub: str, color: str):
    st.markdown(
        f"""
        <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                    padding:1rem 1.1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:0.76rem;font-weight:800;color:#6B8A7A;
                        text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">
                {label}
            </div>
            <div style="font-size:1.65rem;font-weight:900;color:{color};line-height:1;">
                {value}
            </div>
            <div style="font-size:0.78rem;color:#8AA898;margin-top:5px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    bai_header(
        so="1",
        ten="Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa",
        mo_ta="Ước lượng TFP, phân rã tăng trưởng GDP 2020–2025, dự báo GDP Việt Nam 2030",
        cap_do="DỄ",
        tools=["numpy", "pandas", "plotly"],
        thoi_luong="1 tuần",
    )

    r = compute_bai1()
    years = r["years"]
    Y = r["Y"]
    A = r["A"]
    A_bar = r["A_bar"]
    Yhat = r["Yhat"]
    MAPE = r["MAPE"]
    APE = r["APE"]
    max_error_idx = r["max_error_idx"]
    contrib_df = r["contrib_df"]
    dlnY = r["dlnY"]
    forecast_years = r["forecast_years"]
    Y_fore = r["Y_fore"]
    Y2030 = r["Y2030"]
    cagr = r["cagr_2025_2030"]

    info_box(
        "Mô hình: <b>Yₜ = Aₜ · Kₜ<sup>0,33</sup> · Lₜ<sup>0,42</sup> · "
        "Dₜ<sup>0,10</sup> · AIₜ<sup>0,08</sup> · Hₜ<sup>0,07</sup></b>"
        "&nbsp;&nbsp;|&nbsp;&nbsp;Điều kiện lợi suất không đổi: α+β+γ+δ+θ = 1",
        bg="#E8F5E9",
        border=C1,
        icon="📐",
    )

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        _metric_card("Ā - TFP trung bình", f"{A_bar:.4f}", "ước lượng nghịch đảo", C1)
    with k2:
        _metric_card("MAPE dự báo", f"{MAPE:.2f}%", "mô hình khớp khá tốt", BLUE)
    with k3:
        _metric_card("GDP 2030", f"{Y2030:,.0f}", "nghìn tỷ VND", ORANGE)
    with k4:
        _metric_card("CAGR 2025–2030", f"{cagr:.2f}%", "tăng trưởng/năm", PURPLE)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 1.4.1 — TFP A_t
    # ════════════════════════════════════════════════════════
    section_title("Câu 1.4.1 — Ước lượng năng suất nhân tố tổng hợp TFP (Aₜ)", "📊")
    col_chart, col_table = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown(
            """
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                        padding:1.2rem 1.4rem 0.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
                <div style="font-weight:800;font-size:0.95rem;color:#1A2B1F;margin-bottom:3px;">
                    TFP Aₜ theo năm</div>
                <div style="font-size:0.8rem;color:#8AA898;margin-bottom:0.8rem;">
                    Aₜ = Yₜ / (Kₜ<sup>α</sup>·Lₜ<sup>β</sup>·Dₜ<sup>γ</sup>·AIₜ<sup>δ</sup>·Hₜ<sup>θ</sup>)</div>
            """,
            unsafe_allow_html=True,
        )
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=years,
                y=A,
                mode="lines+markers+text",
                line=dict(color=C1, width=3),
                marker=dict(size=10, color=C1, line=dict(width=2, color="#fff")),
                text=[f"{v:.2f}" for v in A],
                textposition="top center",
                textfont=dict(size=10, color=C1),
                fill="tozeroy",
                fillcolor="rgba(26,107,60,0.07)",
                name="TFP Aₜ",
            )
        )
        fig1.add_hline(
            y=A_bar,
            line_dash="dash",
            line_color="#FB8C00",
            line_width=2,
            annotation_text=f"Ā = {A_bar:.4f}",
            annotation_position="right",
        )
        fig1.update_layout(
            height=290,
            margin=dict(l=0, r=60, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=CF,
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor="#F0F4F0", tickfont=dict(size=11), zeroline=False),
            xaxis=dict(showgrid=False, tickfont=dict(size=11), dtick=1),
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_table:
        df_tfp = r["df_tfp"].copy()
        df_tfp["Y thực tế (nghìn tỷ VND)"] = df_tfp["Y thực tế (nghìn tỷ VND)"].map(lambda x: f"{x:,.1f}")
        df_tfp["A_t (TFP)"] = df_tfp["A_t (TFP)"].map(lambda x: f"{x:.4f}")
        df_tfp["Tăng/giảm A_t (%)"] = df_tfp["Tăng/giảm A_t (%)"].map(lambda x: "-" if pd.isna(x) else f"{x:+.2f}%")
        st.dataframe(df_tfp, use_container_width=True, hide_index=True)
        info_box(
            f"TFP tăng từ <b>{A[0]:.2f}</b> lên <b>{A[-1]:.2f}</b> "
            f"(+{(A[-1] / A[0] - 1) * 100:.1f}% cả giai đoạn). "
            "Điều này cho thấy phần hiệu quả còn lại của nền kinh tế có xu hướng cải thiện sau khi đã tính K, L, D, AI và H.",
            bg="#E8F5E9",
            border=C3,
            icon="💡",
        )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 1.4.2 — Yhat và MAPE
    # ════════════════════════════════════════════════════════
    section_title("Câu 1.4.2 — Dự báo Ŷₜ với Ā và tính MAPE", "🎯")
    st.markdown(
        """
        <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                    padding:1.2rem 1.4rem 0.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
            <div style="font-weight:800;font-size:0.95rem;color:#1A2B1F;margin-bottom:3px;">
                Y thực tế và Ŷ dự báo khi dùng Ā = trung bình TFP</div>
            <div style="font-size:0.8rem;color:#8AA898;margin-bottom:0.8rem;">
                MAPE = mean(|Yₜ−Ŷₜ|/Yₜ) × 100</div>
        """,
        unsafe_allow_html=True,
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=years, y=Y, name="Y thực tế", marker_color=C1, opacity=0.85))
    fig2.add_trace(
        go.Scatter(
            x=years,
            y=Yhat,
            name="Ŷ dự báo",
            mode="lines+markers",
            line=dict(color=RED, width=2.6, dash="dot"),
            marker=dict(size=9, color=RED, line=dict(width=2, color="#fff")),
        )
    )
    fig2.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CF,
        bargap=0.3,
        legend=dict(orientation="h", y=-0.22),
        yaxis=dict(showgrid=True, gridcolor="#F0F4F0", tickfont=dict(size=11), zeroline=False, title="Nghìn tỷ VND"),
        xaxis=dict(showgrid=False, tickfont=dict(size=11), dtick=1),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    df_fit = r["df_fit"].copy()
    for col in ["Y thực tế", "Y dự báo", "Sai số tuyệt đối"]:
        df_fit[col] = df_fit[col].map(lambda x: f"{x:,.1f}")
    df_fit["APE (%)"] = df_fit["APE (%)"].map(lambda x: f"{x:.2f}%")
    st.dataframe(df_fit, use_container_width=True, hide_index=True)
    info_box(
        f"<b>MAPE = {MAPE:.2f}%</b>. Sai số lớn nhất nằm ở năm "
        f"<b>{int(years[max_error_idx])}</b> với APE = <b>{APE[max_error_idx]:.2f}%</b>. "
        "Mức MAPE dưới 10% cho thấy mô hình mô tả dữ liệu lịch sử tương đối tốt, dù vẫn có sai số ở hai đầu giai đoạn.",
        bg="#E3F2FD",
        border=BLUE,
        icon="📋",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 1.4.3 — Phân rã tăng trưởng
    # ════════════════════════════════════════════════════════
    section_title("Câu 1.4.3 — Phân rã đóng góp tăng trưởng GDP 2020–2025", "📈")
    st.markdown(
        """
        <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                    padding:1.2rem 1.4rem 0.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
            <div style="font-weight:800;font-size:0.95rem;color:#1A2B1F;margin-bottom:3px;">
                Đóng góp bình quân theo log, đơn vị điểm %/năm</div>
            <div style="font-size:0.8rem;color:#8AA898;margin-bottom:0.8rem;">
                ΔlnY = ΔlnA + αΔlnK + βΔlnL + γΔlnD + δΔlnAI + θΔlnH</div>
        """,
        unsafe_allow_html=True,
    )
    labels = contrib_df["Yếu tố"].tolist()
    values = contrib_df["Đóng góp (điểm %/năm)"].to_numpy()
    colors = [C1, C2, C3, BLUE, PURPLE, "#FB8C00"]
    fig3 = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            hovertemplate="%{x}<br>Đóng góp: %{y:.4f} điểm %/năm<extra></extra>",
        )
    )
    fig3.update_layout(
        height=310,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CF,
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor="#F0F4F0", zeroline=True, title="Điểm %/năm"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10.5)),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    df_contrib_show = contrib_df.copy()
    df_contrib_show["Đóng góp (điểm %/năm)"] = df_contrib_show["Đóng góp (điểm %/năm)"].map(lambda x: f"{x:.4f}")
    df_contrib_show["Tỷ trọng trong tăng trưởng (%)"] = df_contrib_show["Tỷ trọng trong tăng trưởng (%)"].map(lambda x: f"{x:.2f}%")
    st.dataframe(df_contrib_show, use_container_width=True, hide_index=True)

    new_factor_rows = contrib_df[contrib_df["Yếu tố"].isin(["Số hóa (D)", "AI", "Nhân lực số (H)"])].copy()
    top_new = new_factor_rows.sort_values("Đóng góp (điểm %/năm)", ascending=False).iloc[0]
    info_box(
        f"Tăng trưởng GDP bình quân theo log: <b>{dlnY * 100:.3f}%/năm</b>. "
        "Trong toàn bộ mô hình, <b>TFP</b> đóng góp lớn nhất, sau đó là <b>vốn K</b>. "
        f"Trong ba yếu tố mới D, AI, H, yếu tố đóng góp lớn nhất là <b>{top_new['Yếu tố']}</b>.",
        bg="#FFF8E1",
        border="#FB8C00",
        icon="💡",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 1.4.4 — Dự báo 2030
    # ════════════════════════════════════════════════════════
    section_title("Câu 1.4.4 — Dự báo GDP Việt Nam năm 2030", "🚀")
    st.markdown(
        """
        <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                    padding:1.2rem 1.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);margin-bottom:1rem;">
            <div style="font-weight:800;font-size:0.95rem;color:#1A2B1F;margin-bottom:0.8rem;">
                Giả định kịch bản đến năm 2030</div>
        """,
        unsafe_allow_html=True,
    )
    kb1, kb2, kb3 = st.columns(3, gap="medium")
    for col, (lbl, val, sub, color) in zip(
        [kb1, kb2, kb3],
        [
            ("Kinh tế số D", "19,5% → 30%", "tăng dần đến 2030", C1),
            ("AI capacity", "80,1k → 100k DN", "nghìn doanh nghiệp số", BLUE),
            ("Nhân lực số H", "29,2% → 35%", "lao động qua đào tạo", PURPLE),
        ],
    ):
        with col:
            st.markdown(
                f"""
                <div style="background:{color}0D;border:1.5px solid {color}33;border-radius:10px;
                            padding:0.9rem 1rem;text-align:center;">
                    <div style="font-size:0.78rem;font-weight:800;color:{color};text-transform:uppercase;margin-bottom:5px;">
                        {lbl}</div>
                    <div style="font-size:1.1rem;font-weight:900;color:#1A2B1F;">{val}</div>
                    <div style="font-size:0.78rem;color:#8AA898;margin-top:3px;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(
        "<div style='font-size:0.82rem;color:#6B8A7A;margin-top:0.8rem;'>K và L tăng 6%/năm · TFP tăng 1,2%/năm</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                    padding:1.2rem 1.4rem 0.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
            <div style="font-weight:800;font-size:0.95rem;color:#1A2B1F;margin-bottom:3px;">
                Quỹ đạo GDP 2020–2030</div>
            <div style="font-size:0.8rem;color:#8AA898;margin-bottom:0.8rem;">
                Dự báo nối từ GDP thực tế 2025, sau đó mô phỏng đến mục tiêu 2030.</div>
        """,
        unsafe_allow_html=True,
    )
    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=years,
            y=Y,
            name="Lịch sử 2020–2025",
            mode="lines+markers",
            line=dict(color=C1, width=3),
            marker=dict(size=9, color=C1, line=dict(width=2, color="#fff")),
        )
    )
    fig4.add_trace(
        go.Scatter(
            x=forecast_years,
            y=Y_fore,
            name="Dự báo 2025–2030",
            mode="lines+markers",
            line=dict(color=ORANGE, width=3, dash="dash"),
            marker=dict(size=9, color=ORANGE, line=dict(width=2, color="#fff")),
        )
    )
    fig4.add_annotation(
        x=2030,
        y=Y2030,
        text=f"<b>{Y2030:,.0f}</b><br>(+{cagr:.2f}%/năm)",
        showarrow=True,
        arrowhead=2,
        arrowcolor=ORANGE,
        font=dict(size=12, color=ORANGE),
        bgcolor="#FFF3E0",
        bordercolor=ORANGE,
        borderwidth=1,
        ax=-85,
        ay=-35,
    )
    fig4.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CF,
        legend=dict(orientation="h", y=-0.22),
        yaxis=dict(showgrid=True, gridcolor="#F0F4F0", tickfont=dict(size=11), zeroline=False, title="Nghìn tỷ VND"),
        xaxis=dict(showgrid=False, tickfont=dict(size=11), dtick=1),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    df_forecast = r["df_forecast"].copy()
    show_forecast = df_forecast[df_forecast["Năm"].isin([2025, 2026, 2027, 2028, 2029, 2030])].copy()
    show_forecast["GDP dự báo"] = show_forecast["GDP dự báo"].map(lambda x: f"{x:,.1f}")
    for col in ["K", "L", "D", "AI", "H", "A"]:
        show_forecast[col] = show_forecast[col].map(lambda x: f"{x:,.2f}")
    st.dataframe(show_forecast, use_container_width=True, hide_index=True)
    info_box(
        f"<b>GDP 2030 dự báo: {Y2030:,.0f} nghìn tỷ VND</b>; "
        f"CAGR 2025–2030 khoảng <b>{cagr:.2f}%/năm</b>. "
        "Kết quả này phụ thuộc mạnh vào giả định duy trì tăng trưởng K, L, TFP và đạt mục tiêu D = 30% GDP.",
        bg="#E8F5E9",
        border=C1,
        icon="🎯",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Câu hỏi thảo luận chính sách
    # ════════════════════════════════════════════════════════
    section_title("Câu hỏi thảo luận chính sách", "💬")
    d_contrib = contrib_df.set_index("Yếu tố")["Đóng góp (điểm %/năm)"]
    questions = [
        (
            "a)",
            "TFP của Việt Nam có xu hướng tăng hay giảm trong giai đoạn 2020–2025? Điều đó nói lên gì về chất lượng tăng trưởng?",
            f"TFP tăng liên tục từ {A[0]:.2f} lên {A[-1]:.2f}. Điều này cho thấy phần hiệu quả còn lại của nền kinh tế có xu hướng cải thiện. Tuy nhiên, cần diễn giải cẩn trọng: vì mô hình đã đưa D, AI và H vào đầu vào, TFP là phần tăng trưởng chưa được giải thích bởi các biến này, không nên quy hoàn toàn cho số hóa hoặc AI.",
        ),
        (
            "b)",
            "Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp nhiều nhất cho tăng trưởng giai đoạn vừa qua? Vì sao?",
            f"Số hóa D đóng góp nhiều nhất trong ba yếu tố mới, khoảng {d_contrib['Số hóa (D)']:.3f} điểm %/năm; tiếp theo là AI khoảng {d_contrib['AI']:.3f} điểm %/năm và H khoảng {d_contrib['Nhân lực số (H)']:.3f} điểm %/năm. D lớn nhất vì tỷ trọng kinh tế số/GDP tăng nhanh từ 12,0% lên 19,5% trong giai đoạn 2020–2025.",
        ),
        (
            "c)",
            "Mục tiêu Việt Nam đạt 30% kinh tế số/GDP vào 2030 có khả thi không nếu dựa trên mô hình này? Cần ràng buộc gì?",
            f"Theo kịch bản mô phỏng, mục tiêu này có thể hỗ trợ GDP 2030 đạt khoảng {Y2030:,.0f} nghìn tỷ VND. Tuy nhiên, tính khả thi phụ thuộc vào các ràng buộc: duy trì tăng K và L khoảng 6%/năm, TFP tăng 1,2%/năm, mở rộng doanh nghiệp công nghệ số lên 100 nghìn, nâng tỷ lệ lao động qua đào tạo lên 35%, và bảo đảm hạ tầng số, dữ liệu, an toàn thông tin cùng nguồn nhân lực AI.",
        ),
    ]

    for code, q, ans in questions:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()


if __name__ == "__main__":
    render()
