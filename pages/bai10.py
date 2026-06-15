"""pages/bai10.py — Bài 10: Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định

Bản làm lại: bám đúng đề gốc (không tự thêm sàn/trần ngân sách từng hạng mục),
ưu tiên Pyomo nếu có solver GLPK/CBC; fallback bằng scipy.optimize.linprog để luôn chạy được.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import bai_header, end_padding, info_box, section_title

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import pyomo.environ as pyo
    HAS_PYOMO = True
except Exception:
    HAS_PYOMO = False

C1 = "#1A6B3C"; C2 = "#2E8B57"; C3 = "#4CAF72"
CBLUE = "#1976D2"; CRED = "#E53935"; CORANGE = "#E65100"; CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

J = ["I", "D", "AI", "H"]
J_NAMES = {"I":"I — Hạ tầng số", "D":"D — Chuyển đổi số", "AI":"AI — Năng lực AI", "H":"H — Nhân lực số"}
S = ["s1", "s2", "s3", "s4"]
S_NAMES = {"s1":"Lạc quan", "s2":"Cơ sở", "s3":"Bi quan", "s4":"Khủng hoảng"}
S_COLORS = {"s1":C1, "s2":CBLUE, "s3":CORANGE, "s4":CRED}
P = {"s1":0.30, "s2":0.45, "s3":0.20, "s4":0.05}
SCENARIO_TABLE = {
    "s1": {"global_growth":3.5, "fdi":32.0, "export_growth":12.0},
    "s2": {"global_growth":2.8, "fdi":27.0, "export_growth":8.0},
    "s3": {"global_growth":1.5, "fdi":20.0, "export_growth":3.0},
    "s4": {"global_growth":0.2, "fdi":12.0, "export_growth":-5.0},
}

BETA = {"I":1.00, "D":1.10, "AI":1.25, "H":0.95}
BETA_S = {
    ("s1","I"):1.25, ("s1","D"):1.35, ("s1","AI"):1.55, ("s1","H"):1.05,
    ("s2","I"):1.00, ("s2","D"):1.10, ("s2","AI"):1.25, ("s2","H"):0.95,
    ("s3","I"):0.75, ("s3","D"):0.85, ("s3","AI"):0.90, ("s3","H"):1.00,
    ("s4","I"):0.40, ("s4","D"):0.50, ("s4","AI"):0.55, ("s4","H"):1.10,
}
B1, B2 = 65000.0, 15000.0


def _beta_avg():
    return {j: sum(P[s] * BETA_S[(s, j)] for s in S) for j in J}


def _idx_y(s_idx, j_idx):
    return 4 + 4 * s_idx + j_idx


def _solve_sp_scipy():
    """Deterministic equivalent của bài toán SP. Biến: x_j và y_sj."""
    if not HAS_SCIPY:
        return None
    n = 4 + len(S) * 4
    c = np.zeros(n)
    for j_idx, j in enumerate(J):
        c[j_idx] = -BETA[j]
    for s_idx, s in enumerate(S):
        for j_idx, j in enumerate(J):
            c[_idx_y(s_idx, j_idx)] = -P[s] * BETA_S[(s, j)]

    A, b = [], []
    row = np.zeros(n); row[:4] = 1
    A.append(row); b.append(B1)
    for s_idx, _s in enumerate(S):
        row = np.zeros(n); row[4 + 4*s_idx:4 + 4*(s_idx+1)] = 1
        A.append(row); b.append(B2)
        row = np.zeros(n); row[_idx_y(s_idx, 2)] = 1; row[3] = -0.5
        A.append(row); b.append(0)

    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)] * n, method="highs")
    if not res.success:
        return None
    x = {j: float(res.x[i]) for i, j in enumerate(J)}
    y = {(s, j): float(res.x[_idx_y(si, ji)]) for si, s in enumerate(S) for ji, j in enumerate(J)}
    return {"Z": float(-res.fun), "x": x, "y": y, "method":"SciPy/HiGHS"}


def _solve_sp_pyomo():
    """Cài đặt Pyomo đúng cấu trúc Set/Param/Var; trả None nếu thiếu solver."""
    if not HAS_PYOMO:
        return None
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=J)
    m.S = pyo.Set(initialize=S)
    m.p = pyo.Param(m.S, initialize=P)
    m.beta = pyo.Param(m.J, initialize=BETA)
    m.beta_s = pyo.Param(m.S, m.J, initialize=BETA_S)
    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
    m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= B1)
    m.budget2 = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) <= B2)
    m.ai_absorb = pyo.Constraint(m.S, rule=lambda m, s: m.y[s, "AI"] <= 0.5 * m.x["H"])
    m.obj = pyo.Objective(
        expr=sum(m.beta[j] * m.x[j] for j in m.J) +
             sum(m.p[s] * sum(m.beta_s[s, j] * m.y[s, j] for j in m.J) for s in m.S),
        sense=pyo.maximize,
    )
    for solver_name in ["cbc", "glpk"]:
        try:
            solver = pyo.SolverFactory(solver_name)
            if solver is not None and solver.available(False):
                result = solver.solve(m, tee=False)
                status = str(result.solver.termination_condition).lower()
                if "optimal" in status:
                    x = {j: float(pyo.value(m.x[j])) for j in J}
                    y = {(s, j): float(pyo.value(m.y[s, j])) for s in S for j in J}
                    return {"Z": float(pyo.value(m.obj)), "x": x, "y": y, "method":f"Pyomo/{solver_name.upper()}"}
        except Exception:
            pass
    return None


def _solve_ev_scipy():
    """Expected-value deterministic model: thay beta_s bằng beta trung bình."""
    if not HAS_SCIPY:
        return None
    avg = _beta_avg()
    n = 8
    c = np.zeros(n)
    for j_idx, j in enumerate(J):
        c[j_idx] = -BETA[j]
        c[4 + j_idx] = -avg[j]
    A, b = [], []
    row = np.zeros(n); row[:4] = 1
    A.append(row); b.append(B1)
    row = np.zeros(n); row[4:] = 1
    A.append(row); b.append(B2)
    row = np.zeros(n); row[4 + 2] = 1; row[3] = -0.5
    A.append(row); b.append(0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)]*n, method="highs")
    if not res.success:
        return None
    return {
        "Z_ev_model": float(-res.fun),
        "x": {j: float(res.x[i]) for i, j in enumerate(J)},
        "y_mean": {j: float(res.x[4+i]) for i, j in enumerate(J)},
    }


def _solve_recourse_given_x(x, s):
    if not HAS_SCIPY:
        return None
    c = np.array([-BETA_S[(s, j)] for j in J], dtype=float)
    A = [np.ones(4), np.array([0, 0, 1, 0], dtype=float)]
    b = [B2, 0.5 * x["H"]]
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)]*4, method="highs")
    if not res.success:
        return None
    return float(-res.fun), {j: float(res.x[i]) for i, j in enumerate(J)}


def _evaluate_x_stochastic(x):
    first = sum(BETA[j] * x[j] for j in J)
    total = first
    y_eval = {}
    z_s = {}
    for s in S:
        rec = _solve_recourse_given_x(x, s)
        if rec is None:
            return None
        rec_val, y_s = rec
        y_eval[s] = y_s
        z_s[s] = first + rec_val
        total += P[s] * rec_val
    return float(total), y_eval, z_s


def _solve_ws_scipy(s):
    """Wait-and-see: biết trước kịch bản, x và y có thể tối ưu riêng cho kịch bản đó."""
    if not HAS_SCIPY:
        return None
    n = 8
    c = np.zeros(n)
    for j_idx, j in enumerate(J):
        c[j_idx] = -BETA[j]
        c[4 + j_idx] = -BETA_S[(s, j)]
    A, b = [], []
    row = np.zeros(n); row[:4] = 1
    A.append(row); b.append(B1)
    row = np.zeros(n); row[4:] = 1
    A.append(row); b.append(B2)
    row = np.zeros(n); row[4 + 2] = 1; row[3] = -0.5
    A.append(row); b.append(0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)]*n, method="highs")
    if not res.success:
        return None
    return {
        "Z": float(-res.fun),
        "x": {j: float(res.x[i]) for i, j in enumerate(J)},
        "y": {j: float(res.x[4+i]) for i, j in enumerate(J)},
    }


def _solve_robust_regret_scipy(Z_ws):
    """Robust mở rộng: cực tiểu hóa regret kịch bản xấu nhất.

    minimize R
    s.t. R >= Z_WS_s - [beta*x + beta_s*y_s] với mọi s.
    """
    if not HAS_SCIPY:
        return None
    n = 4 + len(S) * 4 + 1
    idx_R = n - 1
    c = np.zeros(n); c[idx_R] = 1
    A, b = [], []
    row = np.zeros(n); row[:4] = 1
    A.append(row); b.append(B1)
    for s_idx, s in enumerate(S):
        row = np.zeros(n); row[4 + 4*s_idx:4 + 4*(s_idx+1)] = 1
        A.append(row); b.append(B2)
        row = np.zeros(n); row[_idx_y(s_idx, 2)] = 1; row[3] = -0.5
        A.append(row); b.append(0)
        # Z_ws_s - benefit_s - R <= 0
        row = np.zeros(n)
        for j_idx, j in enumerate(J):
            row[j_idx] = -BETA[j]
            row[_idx_y(s_idx, j_idx)] = -BETA_S[(s, j)]
        row[idx_R] = -1
        A.append(row); b.append(-Z_ws[s])
    bounds = [(0, None)] * (n-1) + [(0, None)]
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    if not res.success:
        return None
    x = {j: float(res.x[i]) for i, j in enumerate(J)}
    y = {(s, j): float(res.x[_idx_y(si, ji)]) for si, s in enumerate(S) for ji, j in enumerate(J)}
    z_s = {}
    regrets = {}
    first = sum(BETA[j] * x[j] for j in J)
    for s in S:
        benefit = first + sum(BETA_S[(s, j)] * y[(s, j)] for j in J)
        z_s[s] = benefit
        regrets[s] = Z_ws[s] - benefit
    return {"R": float(res.fun), "x": x, "y": y, "z_s": z_s, "regrets": regrets}


def _load_macro_reference():
    """Không tham gia mô hình, chỉ đối chiếu bối cảnh độ mở thương mại từ CSV nếu có."""
    candidates = [
        "data/vietnam_macro_2020_2025.csv",
        "vietnam_macro_2020_2025.csv",
        "/mnt/data/vietnam_macro_2020_2025.csv",
        "/mnt/data/vietnam_macro_2020_2025(1).csv",
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                row = df[df["year"] == 2025].iloc[0]
                openness = (row["exports_billion_USD"] + row["imports_billion_USD"]) / row["GDP_billion_USD"] * 100
                return {
                    "exports": float(row["exports_billion_USD"]),
                    "imports": float(row["imports_billion_USD"]),
                    "gdp": float(row["GDP_billion_USD"]),
                    "openness": float(openness),
                    "path": path,
                }
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def solve_all():
    sp = _solve_sp_pyomo() or _solve_sp_scipy()
    ev = _solve_ev_scipy()
    if sp is None or ev is None:
        return None
    eev = _evaluate_x_stochastic(ev["x"])
    if eev is None:
        return None
    EEV, y_eev, z_eev_s = eev

    ws = {s: _solve_ws_scipy(s) for s in S}
    if any(v is None for v in ws.values()):
        return None
    WS = sum(P[s] * ws[s]["Z"] for s in S)
    EVPI = WS - sp["Z"]
    VSS = sp["Z"] - EEV
    robust = _solve_robust_regret_scipy({s: ws[s]["Z"] for s in S})

    z_sp_s = {}
    first_sp = sum(BETA[j] * sp["x"][j] for j in J)
    for s in S:
        z_sp_s[s] = first_sp + sum(BETA_S[(s, j)] * sp["y"][(s, j)] for j in J)

    return {
        "sp": sp,
        "ev": ev,
        "EEV": EEV,
        "y_eev": y_eev,
        "z_eev_s": z_eev_s,
        "ws": ws,
        "WS": WS,
        "EVPI": EVPI,
        "VSS": VSS,
        "robust": robust,
        "z_sp_s": z_sp_s,
        "macro": _load_macro_reference(),
    }


def _fmt(v, digits=0):
    return f"{v:,.{digits}f}"


def _alloc_df(x):
    return pd.DataFrame({"Hạng mục": [J_NAMES[j] for j in J], "Giá trị": [x[j] for j in J]})


def render():
    bai_header(
        so="10",
        ten="Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định",
        mo_ta="Two-stage stochastic programming, Pyomo + fallback HiGHS, VSS, EVPI và robust regret",
        cap_do="KHÓ",
        tools=["pyomo", "glpk/cbc", "scipy", "plotly"],
        thoi_luong="3 tuần",
    )

    R = solve_all()
    if R is None:
        st.error("Không giải được mô hình. Cần cài scipy hoặc Pyomo + GLPK/CBC.")
        st.code("pip install scipy pyomo\n# Ubuntu: sudo apt-get install glpk-utils coinor-cbc")
        return

    sp = R["sp"]; ev = R["ev"]; ws = R["ws"]; robust = R["robust"]

    info_box(
        "Mô hình đúng theo đề: <b>max Σβⱼxⱼ + ΣpₛΣβⱼˢyⱼˢ</b>. "
        "First-stage x dùng tối đa 65.000 tỷ; second-stage y dùng tối đa 15.000 tỷ/kịch bản. "
        "Ràng buộc cầu nối: <b>y_AI,s ≤ 0,5·x_H</b>. Bản này không tự thêm sàn/trần từng hạng mục ngoài đề.",
        bg="#E8F5E9", border=C1, icon="📐"
    )

    if R["macro"]:
        m = R["macro"]
        info_box(
            f"Đối chiếu bối cảnh bằng CSV macro: năm 2025, xuất khẩu {m['exports']:.1f} tỷ USD, "
            f"nhập khẩu {m['imports']:.1f} tỷ USD, GDP {m['gdp']:.1f} tỷ USD → "
            f"độ mở thương mại ≈ <b>{m['openness']:.1f}% GDP</b>. Dữ liệu này chỉ dùng để kiểm tra bối cảnh, "
            "không đưa vào hàm mục tiêu của Bài 10.",
            bg="#E3F2FD", border=CBLUE, icon="🧾"
        )

    # Scenario table
    section_title("Cây kịch bản và hệ số β", "🌳")
    col_s, col_b = st.columns([2, 3], gap="large")
    with col_s:
        df_s = pd.DataFrame({
            "Kịch bản": [S_NAMES[s] for s in S],
            "Tăng trưởng TG (%)": [SCENARIO_TABLE[s]["global_growth"] for s in S],
            "FDI VN": [SCENARIO_TABLE[s]["fdi"] for s in S],
            "XK tăng (%)": [SCENARIO_TABLE[s]["export_growth"] for s in S],
            "Xác suất": [P[s] for s in S],
        })
        st.dataframe(df_s, use_container_width=True, hide_index=True)
    with col_b:
        df_b = pd.DataFrame({
            "Hạng mục": [J_NAMES[j] for j in J],
            "β cơ bản": [BETA[j] for j in J],
            **{S_NAMES[s]: [BETA_S[(s, j)] for j in J] for s in S},
            "β kỳ vọng": [_beta_avg()[j] for j in J],
        })
        st.dataframe(df_b.round(3), use_container_width=True, hide_index=True)

    info_box(
        "Lý luận quan trọng: trong dữ liệu đề, β_AI cơ bản cao nhất nên first-stage có xu hướng dồn vào AI. "
        "Tuy nhiên do y_AI,s bị chặn bởi x_H, second-stage chỉ đầu tư AI nếu đầu tư H ban đầu đủ lớn. "
        "Nếu lợi ích tăng thêm của y_AI không bù được chi phí cơ hội của x_H, mô hình sẽ không tăng x_H.",
        bg="#FFF8E1", border=CORANGE, icon="💡"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # 10.5.1
    section_title("Câu 10.5.1 — Nghiệm Stochastic Programming", "🎯")
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    kpis = [
        (k1, "Z* SP", _fmt(sp["Z"], 2), "GDP gain kỳ vọng", C1),
        (k2, "Solver", sp["method"], "ưu tiên Pyomo", CBLUE),
        (k3, "First-stage", _fmt(sum(sp["x"].values()), 0), f"/{B1:,.0f} tỷ", CPURPLE),
        (k4, "Recourse/kịch bản", _fmt(B2, 0), "tỷ VND", CORANGE),
    ]
    for col, lbl, val, sub, color in kpis:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:0.78rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
                <div style="font-size:1.55rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                <div style="font-size:0.78rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    col_x, col_y = st.columns([2, 3], gap="large")
    with col_x:
        st.markdown("**First-stage x* — quyết định here-and-now**")
        df_x = _alloc_df(sp["x"])
        st.dataframe(df_x.round(2), use_container_width=True, hide_index=True)
        fig_x = go.Figure(go.Bar(
            x=[J_NAMES[j] for j in J], y=[sp["x"][j] for j in J],
            marker_color=[C3, CBLUE, CRED, CPURPLE], text=[_fmt(sp["x"][j]) for j in J], textposition="outside"
        ))
        fig_x.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0), font=CF,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False, yaxis=dict(showgrid=True, gridcolor="#F0F4F0"))
        st.plotly_chart(fig_x, use_container_width=True, config={"displayModeBar": False})
    with col_y:
        st.markdown("**Second-stage y* — recourse theo từng kịch bản**")
        rows = []
        for s in S:
            rows.append({"Kịch bản": S_NAMES[s], **{j: sp["y"][(s, j)] for j in J}, "Tổng": sum(sp["y"][(s, j)] for j in J)})
        st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)
        fig_y = go.Figure()
        colors = [C3, CBLUE, CRED, CPURPLE]
        for j_idx, j in enumerate(J):
            fig_y.add_trace(go.Bar(
                name=J_NAMES[j], x=[S_NAMES[s] for s in S], y=[sp["y"][(s, j)] for s in S],
                marker_color=colors[j_idx]
            ))
        fig_y.update_layout(height=260, barmode="stack", margin=dict(l=0,r=0,t=10,b=0), font=CF,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", y=-0.25),
                            yaxis=dict(showgrid=True, gridcolor="#F0F4F0"))
        st.plotly_chart(fig_y, use_container_width=True, config={"displayModeBar": False})

    info_box(
        "Kết quả chính xác theo đề là nghiệm góc: x_AI = 65.000 và các x khác bằng 0. "
        "Ở kịch bản tốt, phần dự phòng đi vào D; ở kịch bản xấu, phần dự phòng đi vào H. "
        "Nếu muốn nghiệm đa dạng hơn, cần thêm ràng buộc chính sách như sàn H hoặc giới hạn tập trung, nhưng đề gốc chưa cho.",
        bg="#E3F2FD", border=CBLUE, icon="📋"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # 10.5.2
    section_title("Câu 10.5.2 — EV, WS và so sánh với SP", "⚖️")
    col_cmp, col_ws = st.columns([2, 3], gap="large")
    with col_cmp:
        df_cmp = pd.DataFrame({
            "Hạng mục": [J_NAMES[j] for j in J],
            "x_EV": [ev["x"][j] for j in J],
            "x_SP": [sp["x"][j] for j in J],
        })
        st.dataframe(df_cmp.round(2), use_container_width=True, hide_index=True)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="EV", x=[J_NAMES[j] for j in J], y=[ev["x"][j] for j in J], marker_color=CORANGE))
        fig_cmp.add_trace(go.Bar(name="SP", x=[J_NAMES[j] for j in J], y=[sp["x"][j] for j in J], marker_color=C1))
        fig_cmp.update_layout(height=260, barmode="group", margin=dict(l=0,r=0,t=10,b=0), font=CF,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=-0.25), yaxis=dict(showgrid=True, gridcolor="#F0F4F0"))
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})
    with col_ws:
        rows = []
        for s in S:
            rows.append({
                "Kịch bản": S_NAMES[s],
                "pₛ": P[s],
                "Z_WS,s": ws[s]["Z"],
                "x_WS": ", ".join([f"{j}={ws[s]['x'][j]:.0f}" for j in J if ws[s]["x"][j] > 1e-6]),
                "y_WS": ", ".join([f"{j}={ws[s]['y'][j]:.0f}" for j in J if ws[s]["y"][j] > 1e-6]),
            })
        st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)

    info_box(
        f"EV model có Z_EV = {ev['Z_ev_model']:,.2f}, nhưng khi lấy x_EV đánh giá lại trong mô hình ngẫu nhiên thì "
        f"EEV = {R['EEV']:,.2f}. Vì x_EV và x_SP giống nhau trong dữ liệu đề, VSS bằng 0. Đây là kết quả hợp lý, không phải lỗi solver.",
        bg="#FFF8E1", border=CORANGE, icon="💡"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # 10.5.3
    section_title("Câu 10.5.3 — VSS và EVPI", "💎")
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    vals = [
        (k1, "SP/RP", R["sp"]["Z"], "recourse problem", C1),
        (k2, "EEV", R["EEV"], "EV evaluated", CORANGE),
        (k3, "WS", R["WS"], "perfect info", CBLUE),
        (k4, "VSS", R["VSS"], "SP − EEV", CPURPLE),
        (k5, "EVPI", R["EVPI"], "WS − SP", CRED),
    ]
    for col, lbl, val, sub, color in vals:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:0.9rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);text-align:center;">
                <div style="font-size:0.72rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:5px;">{lbl}</div>
                <div style="font-size:1.35rem;font-weight:900;color:{color};line-height:1;">{val:,.2f}</div>
                <div style="font-size:0.7rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    info_box(
        "VSS = 0 và EVPI = 0 trong bản đề gốc vì quyết định first-stage tối ưu không đổi dù biết hay không biết kịch bản: "
        "đều dồn vào AI do β_AI cơ bản cao nhất. Bài học phương pháp luận: stochastic programming chỉ tạo giá trị rõ rệt "
        "khi bất định làm thay đổi cấu trúc quyết định tối ưu hoặc khi có chi phí/thiếu hụt/penalty kịch bản mạnh hơn.",
        bg="#F3E5F5", border=CPURPLE, icon="🔍"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # 10.5.4
    section_title("Câu 10.5.4 — Robust optimization: cực tiểu hóa regret xấu nhất", "🛡️")
    if robust is not None:
        k1, k2, k3 = st.columns(3, gap="medium")
        for col, lbl, val, sub, color in [
            (k1, "Worst-case regret", robust["R"], "min max regret", CRED),
            (k2, "x_robust AI", robust["x"]["AI"], "tỷ VND", C1),
            (k3, "x_robust H", robust["x"]["H"], "tỷ VND", CPURPLE),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                            padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                    <div style="font-size:0.78rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
                    <div style="font-size:1.7rem;font-weight:900;color:{color};line-height:1;">{val:,.2f}</div>
                    <div style="font-size:0.78rem;color:#8AA898;margin-top:4px;">{sub}</div>
                </div>""", unsafe_allow_html=True)
        df_reg = pd.DataFrame({
            "Kịch bản": [S_NAMES[s] for s in S],
            "Z_WS,s": [ws[s]["Z"] for s in S],
            "Z_robust,s": [robust["z_s"][s] for s in S],
            "Regret": [robust["regrets"][s] for s in S],
        })
        st.dataframe(df_reg.round(2), use_container_width=True, hide_index=True)
        info_box(
            "Ở dữ liệu đề gốc, robust regret cũng trùng với SP và WS theo từng kịch bản, nên regret xấu nhất bằng 0. "
            "Điều này củng cố kết luận: bài toán hiện có bất định về hệ số, nhưng chưa đủ mạnh để làm thay đổi quyết định here-and-now.",
            bg="#E8F5E9", border=C1, icon="🛡️"
        )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    section_title("Câu hỏi thảo luận chính sách", "💬")
    q_data = [
        ("a)", "So với lời giải xác định, SP đầu tư H nhiều hơn hay ít hơn?",
         "Trong dữ liệu đề gốc, SP không đầu tư H ở first-stage, giống EV. Lý do: lợi ích cơ bản của AI (1,25) cao hơn H (0,95), trong khi lợi ích tăng thêm do mở khóa y_AI không đủ bù chi phí cơ hội của x_H. Tuy nhiên, ở recourse kịch bản xấu, mô hình chuyển dự phòng sang H, đúng với vai trò bảo hiểm của nhân lực."),
        ("b)", "VSS dương nói lên điều gì? Vì sao ở đây VSS = 0?",
         "VSS dương nghĩa là tư duy xác suất tạo giá trị: quyết định SP tốt hơn quyết định dựa trên kịch bản trung bình. Ở đây VSS = 0 vì x_EV = x_SP. Đây là kết quả toán học của bộ β hiện tại, không phải lỗi. Muốn VSS dương cần tăng khác biệt kịch bản, thêm penalty khủng hoảng, hoặc đặt sàn khả năng chống chịu."),
        ("c)", "COVID-19/Yagi gợi ý Việt Nam có dưới đầu tư nhân lực số như hàng hóa bảo hiểm không?",
         "Có khả năng. Trong mô hình, H có β cao hơn trong khủng hoảng (1,10), nghĩa là nhân lực số tăng khả năng hấp thụ cú sốc. Nhưng vì xác suất khủng hoảng chỉ 5%, mô hình trung bình vẫn chưa tự tăng x_H. Đây phản ánh một vấn đề thực tế: nếu xác suất rủi ro đuôi bị đánh giá thấp, chính sách dễ dưới đầu tư vào năng lực chống chịu như nhân lực số, dữ liệu dự phòng và an ninh mạng."),
    ]
    for code, q, ans in q_data:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()
