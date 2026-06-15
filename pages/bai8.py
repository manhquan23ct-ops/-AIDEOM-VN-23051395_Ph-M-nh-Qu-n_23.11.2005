"""pages/bai8.py — Bài 8: Tối ưu động phân bổ liên thời gian 2026–2035

Bản làm lại: dùng dữ liệu macro CSV để hiệu chỉnh A0, giải bài toán động phi tuyến
bằng scipy.optimize.SLSQP, có tái tối ưu khi cú sốc 2028 và so sánh chiến lược.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

try:
    from utils import bai_header, end_padding, info_box, section_title
except Exception:  # fallback để file vẫn chạy độc lập khi thiếu utils.py
    def bai_header(so, ten, mo_ta, cap_do, tools, thoi_luong):
        st.title(f"Bài {so}. {ten}")
        st.caption(f"{mo_ta} · Cấp độ: {cap_do} · Công cụ: {', '.join(tools)} · {thoi_luong}")

    def info_box(text, bg="#F7F7F7", border="#DDDDDD", icon="ℹ️"):
        st.markdown(
            f"""
            <div style="background:{bg};border-left:4px solid {border};border-radius:10px;
                        padding:0.9rem 1rem;margin:0.5rem 0 1rem;">
                <div style="font-size:0.92rem;line-height:1.55;">{icon} {text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def section_title(title, icon=""):
        st.markdown(f"### {icon} {title}")

    def end_padding():
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"; C2 = "#2E8B57"; C3 = "#4CAF72"
CBLUE = "#1976D2"; CRED = "#E53935"; CORANGE = "#E65100"; CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

# ─────────────────────────────────────────────────────────────────────────────
# Tham số mô hình theo đề
# ─────────────────────────────────────────────────────────────────────────────
YEARS = np.arange(2026, 2036)
T = len(YEARS)

ALPHA, BETA_L, GAMMA_D, DELTA_AI, THETA_H = 0.33, 0.42, 0.10, 0.08, 0.07
DELTA_K, DELTA_D, DELTA_AI_DEP = 0.05, 0.12, 0.15
THETA_H_ACC, MU_BRAIN = 0.80, 0.02
PHI_D, PHI_AI, PHI_H = 0.003, 0.002, 0.004
RHO_DEFAULT = 0.97
CRRA_GAMMA = 1.5

# Điều kiện ban đầu 2026 theo đề. A0 sẽ được hiệu chỉnh từ Bài 1 nếu đọc được macro CSV.
DEFAULT_INIT = dict(K0=27_500.0, L0=53.9, D0=20.3, AI0=86.0, H0=30.0, A0=34.9136)

# Lưu ý đơn vị:
# K là nghìn tỷ VND, nhưng D/AI/H là chỉ số/tỷ lệ. Nếu cộng trực tiếp I_D, I_AI, I_H
# theo nghìn tỷ vào D/AI/H thì mô hình bùng nổ sai đơn vị. Vì vậy ta dùng hệ số chuyển đổi
# tiền tệ -> điểm chỉ số hiệu quả. Đây là giả định kỹ thuật để làm bài toán động có nghĩa kinh tế.
ETA_K = 1.0       # 1 nghìn tỷ đầu tư K làm tăng 1 nghìn tỷ vốn vật chất
ETA_D = 0.010     # 1 nghìn tỷ đầu tư D làm tăng 0.010 điểm chỉ số D hiệu quả
ETA_AI = 0.008    # 1 nghìn tỷ đầu tư AI làm tăng 0.008 nghìn DN/năng lực AI hiệu quả
ETA_H = 0.010     # 1 nghìn tỷ đầu tư H làm tăng 0.010 điểm vốn nhân lực trước hiệu suất theta_H

# Ràng buộc chính sách bổ sung để nghiệm SLSQP ổn định và có ý nghĩa:
# tổng đầu tư mỗi năm không vượt quá 45% sản lượng, tiêu dùng còn ít nhất 55%.
# Nếu bỏ ràng buộc này, nghiệm hữu hạn vẫn tồn tại do log(C), nhưng rất dễ rơi vào cực trị địa phương cực đoan.
S_MAX = 0.45
S_ITEM_MAX = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# Helper dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
def _candidate_paths(filename: str) -> List[Path]:
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    return [
        here / filename,
        here / "data" / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        Path("/mnt/data") / filename,
        Path("/mnt/data") / filename.replace(".csv", "(1).csv"),
        Path("/mnt/data/final_checked_bai1_7/data") / filename,
    ]


def _read_csv_safely(filename: str) -> Optional[pd.DataFrame]:
    for path in _candidate_paths(filename):
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_initial_conditions() -> Tuple[Dict[str, float], str, pd.DataFrame]:
    """Đọc macro CSV để hiệu chỉnh TFP A0 từ Bài 1.

    CSV macro không có K, AI, H nên các biến này lấy theo bảng đề/Bài 1.
    GDP và D năm 2025 được đọc từ CSV để tính lại A_2025, sau đó cập nhật một bước
    theo cơ chế TFP nội sinh để lấy A0 năm 2026.
    """
    init = DEFAULT_INIT.copy()
    df = _read_csv_safely("vietnam_macro_2020_2025.csv")
    source = "Fallback theo đề vì không tìm thấy vietnam_macro_2020_2025.csv"

    if df is not None and {"year", "GDP_trillion_VND", "digital_economy_share_GDP_pct"}.issubset(df.columns):
        row_2025 = df.loc[df["year"] == 2025]
        if not row_2025.empty:
            Y25 = float(row_2025["GDP_trillion_VND"].iloc[0])
            D25 = float(row_2025["digital_economy_share_GDP_pct"].iloc[0])
            # Tham số 2025 từ bảng Bài 1 trong đề.
            K25, L25, AI25, H25 = 25_900.0, 53.4, 80.1, 29.2
            A25 = Y25 / (K25**ALPHA * L25**BETA_L * D25**GAMMA_D * AI25**DELTA_AI * H25**THETA_H)
            A26 = A25 * (1 + PHI_D * D25/100 + PHI_AI * AI25/100 + PHI_H * H25/100)
            init["A0"] = float(A26)
            source = (
                "Đã đọc vietnam_macro_2020_2025.csv: dùng GDP và D năm 2025 để tính A2025, "
                "sau đó cập nhật một bước TFP nội sinh sang 2026."
            )
    if df is None:
        df = pd.DataFrame()
    return init, source, df


# ─────────────────────────────────────────────────────────────────────────────
# Mô phỏng và tối ưu
# ─────────────────────────────────────────────────────────────────────────────
def labor_path(t: int, L0: float) -> float:
    return L0 * (1.005 ** t)  # giả định lao động tăng 0,5%/năm


def utility(C: np.ndarray, kind: str = "log", crra_gamma: float = CRRA_GAMMA) -> np.ndarray:
    C = np.maximum(np.asarray(C, dtype=float), 1e-9)
    if kind == "crra":
        if abs(crra_gamma - 1.0) < 1e-10:
            return np.log(C)
        return (C ** (1 - crra_gamma)) / (1 - crra_gamma)
    return np.log(C)


def simulate_policy(
    shares: np.ndarray,
    init: Dict[str, float],
    rho: float = RHO_DEFAULT,
    utility_kind: str = "log",
    shock_year: Optional[int] = None,
    shock_pct: float = 0.0,
) -> Tuple[pd.DataFrame, float]:
    """Mô phỏng trạng thái với ma trận shares shape (T,4): [sK,sD,sAI,sH]."""
    S = np.asarray(shares, dtype=float).reshape(T, 4)
    K, D, AI, H, A, L0 = init["K0"], init["D0"], init["AI0"], init["H0"], init["A0"], init["L0"]
    rows = []
    welfare = 0.0

    for t, year in enumerate(YEARS):
        L = labor_path(t, L0)
        # Các trạng thái phải dương để Cobb-Douglas xác định.
        K_eff, D_eff, AI_eff, H_eff = max(K, 1e-6), max(D, 1e-6), max(AI, 1e-6), max(H, 1e-6)
        Y_plan = A * (K_eff**ALPHA) * (L**BETA_L) * (D_eff**GAMMA_D) * (AI_eff**DELTA_AI) * (H_eff**THETA_H)
        shock_factor = 1.0
        if shock_year is not None and int(year) == int(shock_year):
            shock_factor = 1.0 - shock_pct
        Y = Y_plan * shock_factor

        # Chuẩn hóa mềm nếu tổng share vượt S_MAX do sai số optimizer.
        s = np.maximum(S[t], 0.0)
        if s.sum() > S_MAX:
            s = s / s.sum() * S_MAX

        IK, ID, IAI, IH = Y * s
        C = Y - (IK + ID + IAI + IH)
        if C <= 0 or not np.isfinite(C):
            # Gán tiêu dùng nhỏ để tránh log lỗi; nghiệm này sẽ bị objective phạt mạnh.
            C = 1e-9

        rows.append(dict(
            year=int(year), Y_plan=Y_plan, shock_factor=shock_factor, Y=Y, C=C,
            s_K=s[0], s_D=s[1], s_AI=s[2], s_H=s[3], s_total=s.sum(),
            I_K=IK, I_D=ID, I_AI=IAI, I_H=IH,
            K=K, D=D, AI=AI, H=H, A=A, L=L,
        ))
        welfare += (rho ** t) * float(utility(np.array([C]), utility_kind)[0])

        # Chuyển trạng thái. D/AI/H dùng hệ số chuyển đổi tiền tệ -> chỉ số hiệu quả.
        K = (1 - DELTA_K) * K + ETA_K * IK
        D = (1 - DELTA_D) * D + ETA_D * ID
        AI = (1 - DELTA_AI_DEP) * AI + ETA_AI * IAI
        H = (1 - MU_BRAIN) * H + THETA_H_ACC * ETA_H * IH
        A = A * (1 + PHI_D * (D/100) + PHI_AI * (AI/100) + PHI_H * (H/100))

    return pd.DataFrame(rows), float(welfare)


def _policy_from_total_share(total_share: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if weights is None:
        weights = np.array([ALPHA, GAMMA_D, DELTA_AI, THETA_H], dtype=float)
    weights = weights / weights.sum()
    return np.outer(np.asarray(total_share, dtype=float), weights)


def heuristic_policy(kind: str = "even") -> np.ndarray:
    if kind == "front":
        total = np.array([0.42, 0.40, 0.38, 0.34, 0.30, 0.26, 0.22, 0.18, 0.14, 0.10])
    elif kind == "back":
        total = np.array([0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.40, 0.42])
    else:
        total = np.ones(T) * 0.28
    return _policy_from_total_share(np.minimum(total, S_MAX))


@st.cache_data(show_spinner=False)
def solve_dynamic_problem(
    rho: float = RHO_DEFAULT,
    utility_kind: str = "log",
    shock_year: Optional[int] = None,
    shock_pct: float = 0.0,
) -> Dict[str, object]:
    init, source, _ = load_initial_conditions()

    if not HAS_SCIPY:
        shares = heuristic_policy("even")
        df, W = simulate_policy(shares, init, rho, utility_kind, shock_year, shock_pct)
        return dict(success=False, message="Thiếu scipy; dùng chính sách heuristic", shares=shares, df=df, welfare=W, init=init, source=source)

    # Nhiều điểm bắt đầu để giảm rủi ro kẹt cực trị địa phương.
    starts = [
        heuristic_policy("front"),
        heuristic_policy("even"),
    ]

    bounds = [(0.0, S_ITEM_MAX)] * (T * 4)
    constraints = [
        {"type": "ineq", "fun": lambda z, tt=tt: S_MAX - np.reshape(z, (T, 4))[tt].sum()}
        for tt in range(T)
    ]

    def objective(z: np.ndarray) -> float:
        S = np.reshape(z, (T, 4))
        if np.any(S < -1e-9) or np.any(S.sum(axis=1) > S_MAX + 1e-7):
            return 1e9
        df, W = simulate_policy(S, init, rho, utility_kind, shock_year, shock_pct)
        if not np.isfinite(W):
            return 1e9
        return -W

    best = None
    for x0 in starts:
        res = minimize(
            objective,
            x0.reshape(-1),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 350, "ftol": 1e-7, "disp": False},
        )
        if best is None or res.fun < best.fun:
            best = res

    shares = np.reshape(best.x, (T, 4))
    shares = np.maximum(shares, 0.0)
    # Chuẩn hóa lại các năm vượt S_MAX do sai số số học.
    for t in range(T):
        if shares[t].sum() > S_MAX:
            shares[t] *= S_MAX / shares[t].sum()
    df, W = simulate_policy(shares, init, rho, utility_kind, shock_year, shock_pct)
    return dict(success=bool(best.success), message=str(best.message), shares=shares, df=df, welfare=W, init=init, source=source)


# ─────────────────────────────────────────────────────────────────────────────
# Render Streamlit
# ─────────────────────────────────────────────────────────────────────────────
def render():
    bai_header(
        so="8",
        ten="Tối ưu động phân bổ liên thời gian 2026–2035",
        mo_ta="Ramsey–Cobb-Douglas, tích lũy K/D/AI/H, TFP nội sinh, SLSQP và phân tích cú sốc",
        cap_do="KHÁ KHÓ",
        tools=["scipy.optimize", "numpy", "plotly"],
        thoi_luong="2 tuần",
    )

    init, source_note, macro_df = load_initial_conditions()

    info_box(
        "Mục tiêu: <b>max Σ ρᵗ·U(Cₜ)</b>, với U(C)=ln(C) mặc định. "
        "Sản lượng: <b>Yₜ=AₜKₜ<sup>0.33</sup>Lₜ<sup>0.42</sup>Dₜ<sup>0.10</sup>AIₜ<sup>0.08</sup>Hₜ<sup>0.07</sup></b>. "
        "Do bài toán phi tuyến và có động học TFP, bản này giải bằng <b>scipy.optimize.SLSQP</b>.",
        bg="#E8F5E9", border=C1, icon="📐",
    )
    info_box(
        f"<b>Dữ liệu đầu vào:</b> {source_note}<br>"
        f"Điều kiện 2026: K₀={init['K0']:,.0f}, L₀={init['L0']:.1f}, D₀={init['D0']:.1f}, "
        f"AI₀={init['AI0']:.1f}, H₀={init['H0']:.1f}, A₀={init['A0']:.4f}.",
        bg="#E3F2FD", border=CBLUE, icon="🗂️",
    )
    info_box(
        "<b>Lưu ý chặt chẽ về đơn vị:</b> K đo bằng nghìn tỷ VND, còn D/AI/H là chỉ số hoặc tỷ lệ. "
        "Nếu cộng trực tiếp đầu tư tiền tệ vào D/AI/H, mô hình sẽ bùng nổ sai đơn vị. Vì vậy code dùng "
        "hệ số chuyển đổi ηD, ηAI, ηH để biến đầu tư tiền tệ thành điểm chỉ số hiệu quả. Đây là giả định kỹ thuật "
        "cần nêu rõ khi thuyết minh.",
        bg="#FFF8E1", border=CORANGE, icon="⚠️",
    )

    # ── Solve base ───────────────────────────────────────────────────────────
    with st.spinner("Đang giải bài toán động bằng SLSQP..."):
        base = solve_dynamic_problem(rho=RHO_DEFAULT, utility_kind="log")
        shock = solve_dynamic_problem(rho=RHO_DEFAULT, utility_kind="log", shock_year=2028, shock_pct=0.08)
        rho90 = solve_dynamic_problem(rho=0.90, utility_kind="log")

    df_base: pd.DataFrame = base["df"]
    df_shock: pd.DataFrame = shock["df"]
    W_base = float(base["welfare"])
    W_shock = float(shock["welfare"])
    cagr = ((df_base["Y"].iloc[-1] / df_base["Y"].iloc[0]) ** (1/(T-1)) - 1) * 100
    avg_s = df_base["s_total"].mean() * 100

    # ════════════════════════════════════════════════════════
    # 8.3.1 — Solution
    # ════════════════════════════════════════════════════════
    section_title("Câu 8.3.1 — Nghiệm tối ưu động bằng SLSQP", "🎯")

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    kpis = [
        (k1, "Solver", "OK" if base["success"] else "Local", str(base["message"])[:40], C1 if base["success"] else CORANGE),
        (k2, "Welfare", f"{W_base:.4f}", "Σρᵗ·ln(Cₜ)", C1),
        (k3, "Y năm 2035", f"{df_base['Y'].iloc[-1]:,.0f}", "nghìn tỷ VND", CBLUE),
        (k4, "Đầu tư TB", f"{avg_s:.1f}%", "của Y mỗi năm", CPURPLE),
    ]
    for col, lbl, val, sub, color in kpis:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:0.78rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
                <div style="font-size:1.55rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                <div style="font-size:0.75rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    df_view = df_base[["year", "Y", "C", "s_total", "s_K", "s_D", "s_AI", "s_H", "K", "D", "AI", "H", "A"]].copy()
    for c in ["Y", "C", "K"]:
        df_view[c] = df_view[c].map(lambda v: f"{v:,.0f}")
    for c in ["s_total", "s_K", "s_D", "s_AI", "s_H"]:
        df_view[c] = df_view[c].map(lambda v: f"{100*v:.1f}%")
    for c in ["D", "AI", "H", "A"]:
        df_view[c] = df_view[c].map(lambda v: f"{v:.2f}")
    df_view = df_view.rename(columns={
        "year": "Năm", "Y": "Y", "C": "C", "s_total": "Tỷ lệ đầu tư", "s_K": "sK", "s_D": "sD",
        "s_AI": "sAI", "s_H": "sH", "K": "K", "D": "D", "AI": "AI", "H": "H", "A": "TFP A",
    })
    st.dataframe(df_view, use_container_width=True, hide_index=True)
    info_box(
        f"CAGR Y giai đoạn 2026–2035 khoảng <b>{cagr:.2f}%/năm</b>. Vì horizon hữu hạn, nghiệm tối ưu có xu hướng "
        "đầu tư nhiều hơn ở các năm đầu–giữa và giảm dần gần cuối kỳ để bảo toàn tiêu dùng C.",
        bg="#F1F8F2", border=C2, icon="📋",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # 8.3.2 — Trajectories
    # ════════════════════════════════════════════════════════
    section_title("Câu 8.3.2 — Quỹ đạo tối ưu K, D, AI, H, Y, C", "📈")
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Y và C", "Vốn vật chất K", "Số hóa D", "AI", "Nhân lực H", "TFP A"],
        vertical_spacing=0.18, horizontal_spacing=0.08,
    )
    trace_specs = [
        (1, 1, "Y", "Y", C1, "solid"),
        (1, 1, "C", "C", CORANGE, "dot"),
        (1, 2, "K", "K", CBLUE, "solid"),
        (1, 3, "D", "D", CPURPLE, "solid"),
        (2, 1, "AI", "AI", CRED, "solid"),
        (2, 2, "H", "H", C2, "solid"),
        (2, 3, "A", "TFP A", CORANGE, "solid"),
    ]
    for row, col, series, name, color, dash in trace_specs:
        fig.add_trace(go.Scatter(
            x=df_base["year"], y=df_base[series], mode="lines+markers", name=name,
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=6, color=color, line=dict(width=1.5, color="#fff")),
        ), row=row, col=col)
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=45, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CF, showlegend=False)
    for i in range(1, 3):
        for j in range(1, 4):
            fig.update_xaxes(showgrid=False, row=i, col=j)
            fig.update_yaxes(showgrid=True, gridcolor="#F0F4F0", row=i, col=j)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Investment shares plot
    fig_s = go.Figure()
    for col, name, color in [("s_K", "K", CBLUE), ("s_D", "D", CPURPLE), ("s_AI", "AI", CRED), ("s_H", "H", C2)]:
        fig_s.add_trace(go.Bar(x=df_base["year"], y=100*df_base[col], name=name, marker_color=color))
    fig_s.update_layout(
        title="Cơ cấu tỷ lệ đầu tư tối ưu theo năm (% Y)", height=300,
        margin=dict(l=0, r=0, t=35, b=0), barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CF,
        legend=dict(orientation="h", y=-0.25),
        yaxis=dict(title="% sản lượng", showgrid=True, gridcolor="#F0F4F0"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # 8.3.3 — Shock
    # ════════════════════════════════════════════════════════
    section_title("Câu 8.3.3 — Cú sốc năm 2028: Y giảm 8%", "⚡")
    y28_base = float(df_base.loc[df_base["year"] == 2028, "Y"].iloc[0])
    y28_shock = float(df_shock.loc[df_shock["year"] == 2028, "Y"].iloc[0])
    deltaW = W_shock - W_base

    k1, k2, k3 = st.columns(3, gap="medium")
    shock_kpis = [
        (k1, "Y2028 sau cú sốc", f"{y28_shock:,.0f}", f"giảm {y28_base-y28_shock:,.0f}", CRED),
        (k2, "Δ Welfare", f"{deltaW:.4f}", "so với baseline", CORANGE),
        (k3, "Đầu tư TB sau sốc", f"{100*df_shock['s_total'].mean():.1f}%", "tái tối ưu", C1),
    ]
    for col, lbl, val, sub, color in shock_kpis:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:0.78rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
                <div style="font-size:1.55rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                <div style="font-size:0.75rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_base["year"], y=df_base["Y"], mode="lines+markers", name="Baseline",
                              line=dict(color=C1, width=2.5), marker=dict(size=7, color=C1, line=dict(width=1.5, color="#fff"))))
    fig2.add_trace(go.Scatter(x=df_shock["year"], y=df_shock["Y"], mode="lines+markers", name="Tái tối ưu khi sốc",
                              line=dict(color=CRED, width=2.5, dash="dash"), marker=dict(size=7, color=CRED, line=dict(width=1.5, color="#fff"))))
    fig2.add_vrect(x0=2027.5, x1=2028.5, fillcolor=CRED, opacity=0.08, line_width=0)
    fig2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CF,
                       legend=dict(orientation="h", y=-0.25),
                       yaxis=dict(title="Y", showgrid=True, gridcolor="#F0F4F0"), xaxis=dict(showgrid=False))
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    df_delta = pd.DataFrame({
        "Năm": YEARS,
        "ΔsK (điểm %)": 100*(df_shock["s_K"].values - df_base["s_K"].values),
        "ΔsD (điểm %)": 100*(df_shock["s_D"].values - df_base["s_D"].values),
        "ΔsAI (điểm %)": 100*(df_shock["s_AI"].values - df_base["s_AI"].values),
        "ΔsH (điểm %)": 100*(df_shock["s_H"].values - df_base["s_H"].values),
    })
    st.dataframe(df_delta.round(2), use_container_width=True, hide_index=True)
    info_box(
        "Cú sốc làm Y và C năm 2028 giảm trực tiếp. Vì bài toán được tái tối ưu, mô hình điều chỉnh cơ cấu đầu tư "
        "ở các năm sau: hạng mục nào có tác động tích lũy tốt hơn sẽ được giữ/ưu tiên, còn đầu tư kém hiệu quả ở cuối kỳ bị cắt.",
        bg="#FCE4EC", border=CRED, icon="⚡",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # 8.3.4 — Strategies
    # ════════════════════════════════════════════════════════
    section_title("Câu 8.3.4 — So sánh đầu tư đều, front-load và tối ưu", "⚖️")
    even_S = heuristic_policy("even")
    front_S = heuristic_policy("front")
    df_even, W_even = simulate_policy(even_S, init, rho=RHO_DEFAULT)
    df_front, W_front = simulate_policy(front_S, init, rho=RHO_DEFAULT)

    strategy_rows = pd.DataFrame([
        {"Chiến lược": "Tối ưu SLSQP", "Welfare": W_base, "Y2035": df_base["Y"].iloc[-1], "C2035": df_base["C"].iloc[-1], "Đầu tư TB": df_base["s_total"].mean()},
        {"Chiến lược": "Đầu tư trải đều", "Welfare": W_even, "Y2035": df_even["Y"].iloc[-1], "C2035": df_even["C"].iloc[-1], "Đầu tư TB": df_even["s_total"].mean()},
        {"Chiến lược": "Front-load", "Welfare": W_front, "Y2035": df_front["Y"].iloc[-1], "C2035": df_front["C"].iloc[-1], "Đầu tư TB": df_front["s_total"].mean()},
    ]).sort_values("Welfare", ascending=False)

    df_strategy_view = strategy_rows.copy()
    df_strategy_view["Welfare"] = df_strategy_view["Welfare"].map(lambda v: f"{v:.4f}")
    df_strategy_view["Y2035"] = df_strategy_view["Y2035"].map(lambda v: f"{v:,.0f}")
    df_strategy_view["C2035"] = df_strategy_view["C2035"].map(lambda v: f"{v:,.0f}")
    df_strategy_view["Đầu tư TB"] = df_strategy_view["Đầu tư TB"].map(lambda v: f"{100*v:.1f}%")
    st.dataframe(df_strategy_view, use_container_width=True, hide_index=True)

    fig3 = go.Figure()
    for df_, name, color, dash in [(df_base, "Tối ưu", C1, "solid"), (df_even, "Trải đều", CBLUE, "dot"), (df_front, "Front-load", CORANGE, "dash")]:
        fig3.add_trace(go.Scatter(x=df_["year"], y=df_["Y"], mode="lines+markers", name=name,
                                  line=dict(color=color, width=2.5, dash=dash),
                                  marker=dict(size=6, color=color, line=dict(width=1.5, color="#fff"))))
    fig3.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CF,
                       legend=dict(orientation="h", y=-0.25),
                       yaxis=dict(title="Y", showgrid=True, gridcolor="#F0F4F0"), xaxis=dict(showgrid=False))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    best_strategy = strategy_rows.iloc[0]["Chiến lược"]
    info_box(
        f"Chiến lược có welfare cao nhất là <b>{best_strategy}</b>. Front-load thường làm Y cuối kỳ cao hơn, "
        "nhưng có thể giảm welfare nếu hi sinh quá nhiều tiêu dùng đầu kỳ. Đây là đúng bản chất Ramsey: tối ưu không chỉ tối đa hóa Y, "
        "mà tối đa hóa phúc lợi tiêu dùng đã chiết khấu.",
        bg="#FFF8E1", border=CORANGE, icon="💡",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Rho sensitivity
    # ════════════════════════════════════════════════════════
    section_title("Bonus — Độ nhạy khi ρ giảm từ 0,97 xuống 0,90", "🔍")
    df_rho90: pd.DataFrame = rho90["df"]
    rho_rows = pd.DataFrame([
        {"ρ": 0.97, "Welfare": W_base, "Y2035": df_base["Y"].iloc[-1], "Đầu tư TB": df_base["s_total"].mean(), "Đầu tư 2026": df_base["s_total"].iloc[0]},
        {"ρ": 0.90, "Welfare": rho90["welfare"], "Y2035": df_rho90["Y"].iloc[-1], "Đầu tư TB": df_rho90["s_total"].mean(), "Đầu tư 2026": df_rho90["s_total"].iloc[0]},
    ])
    df_rho_view = rho_rows.copy()
    df_rho_view["Welfare"] = df_rho_view["Welfare"].map(lambda v: f"{v:.4f}")
    df_rho_view["Y2035"] = df_rho_view["Y2035"].map(lambda v: f"{v:,.0f}")
    for c in ["Đầu tư TB", "Đầu tư 2026"]:
        df_rho_view[c] = df_rho_view[c].map(lambda v: f"{100*v:.1f}%")
    st.dataframe(df_rho_view, use_container_width=True, hide_index=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_base["year"], y=100*df_base["s_total"], mode="lines+markers", name="ρ=0.97",
                              line=dict(color=C1, width=2.5)))
    fig4.add_trace(go.Scatter(x=df_rho90["year"], y=100*df_rho90["s_total"], mode="lines+markers", name="ρ=0.90",
                              line=dict(color=CRED, width=2.5, dash="dash")))
    fig4.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CF,
                       legend=dict(orientation="h", y=-0.25),
                       yaxis=dict(title="Tỷ lệ đầu tư (%Y)", showgrid=True, gridcolor="#F0F4F0"), xaxis=dict(showgrid=False))
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Discussion
    # ════════════════════════════════════════════════════════
    section_title("Câu hỏi thảo luận chính sách", "💬")
    # Front/back diagnosis
    early_inv = df_base.loc[df_base["year"].between(2026, 2028), "s_total"].mean()
    late_inv = df_base.loc[df_base["year"].between(2033, 2035), "s_total"].mean()
    pattern = "front-loaded" if early_inv > late_inv + 0.03 else ("back-loaded" if late_inv > early_inv + 0.03 else "tương đối phẳng")
    ratio_ai_h = (df_base["I_AI"].sum() / max(df_base["I_H"].sum(), 1e-9))

    q_data = [
        ("a)", "Quỹ đạo tối ưu của K, D, AI, H có front-loaded hay back-loaded không?",
         f"Nghiệm có dạng <b>{pattern}</b>: tỷ lệ đầu tư bình quân 2026–2028 là {100*early_inv:.1f}% Y, "
         f"trong khi 2033–2035 là {100*late_inv:.1f}% Y. Lý do là đầu tư sớm có nhiều năm để tích lũy qua K, D, AI, H và làm tăng TFP; "
         "còn đầu tư quá muộn ít kịp tạo phúc lợi trong horizon hữu hạn."),
        ("b)", "Tỷ lệ đầu tư AI/đầu tư H theo thời gian có ổn định không? Đào tạo nên đi trước hay đồng thời với AI?",
         f"Tổng I_AI/I_H của nghiệm khoảng <b>{ratio_ai_h:.2f}</b>, nhưng theo năm không hoàn toàn ổn định. "
         "Điều này hàm ý không nên chỉ bơm AI độc lập: đầu tư H cần đi song song để tăng năng lực hấp thụ và giảm brain drain; "
         "nếu thiếu H, tác động của AI dễ bị nghẽn ở triển khai."),
        ("c)", "Nếu ρ = 0,90 thì kết quả thay đổi thế nào? Đây có phải lý do chính phủ dưới đầu tư R&D?",
         "Khi ρ thấp hơn, phúc lợi tương lai bị chiết khấu mạnh hơn, nên mô hình có xu hướng bảo vệ tiêu dùng hiện tại và giảm động cơ đầu tư dài hạn. "
         "Đây là một cách lý giải hiện tượng dưới đầu tư vào R&D/hạ tầng số: lợi ích nằm xa trong tương lai, còn chi phí ngân sách và áp lực chính trị xuất hiện ngay."),
    ]
    for code, q, ans in q_data:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()


# Cho phép chạy độc lập: streamlit run bai8.py
if __name__ == "__main__":
    render()
