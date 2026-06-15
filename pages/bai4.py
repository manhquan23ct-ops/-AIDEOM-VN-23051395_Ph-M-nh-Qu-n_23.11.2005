"""pages/bai4.py — Bài 4: Quy hoạch tuyến tính phân bổ ngân sách số theo ngành-vùng

Bản làm lại / đã kiểm tra:
- Cài mô hình bằng PuLP nếu môi trường có PuLP.
- Cài mô hình lại bằng CVXPY nếu môi trường có CVXPY.
- Có fallback bằng scipy.optimize.linprog để app vẫn chạy được khi thiếu PuLP/CVXPY.
- Quan trọng: với đúng tham số đề bài λ=0.7, γ=0.002, trần vùng=12.000,
  bài toán CÓ ràng buộc công bằng C5 là VÔ NGHIỆM. File này hiển thị rõ lý do,
  thay vì dùng nghiệm fallback sai.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import bai_header, end_padding, info_box, section_title

try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình hiển thị
# ─────────────────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"
C2 = "#2E8B57"
C3 = "#4CAF72"
CBLUE = "#1976D2"
CRED = "#E53935"
CORANGE = "#E65100"
CPURPLE = "#7B1FA2"
CGREY = "#6B8A7A"
CF = dict(family="Montserrat, sans-serif", size=12)

# ─────────────────────────────────────────────────────────────────────────────
# Dữ liệu đề bài
# ─────────────────────────────────────────────────────────────────────────────
REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
RNAMES = [
    "Trung du MN Bắc",
    "ĐB sông Hồng",
    "Bắc TB + DHMT",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "ĐB sông Cửu Long",
]
ITEMS = ["I", "D", "AI", "H"]
INAMES = ["I — Hạ tầng số", "D — CĐS DN", "AI — Năng lực AI", "H — Nhân lực số"]

BETA_TABLE = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
], dtype=float)
BETA = {(r, j): float(BETA_TABLE[i, k]) for i, r in enumerate(REGIONS) for k, j in enumerate(ITEMS)}
D0_ARRAY = np.array([38, 78, 55, 32, 82, 48], dtype=float)
D0 = {r: float(D0_ARRAY[i]) for i, r in enumerate(REGIONS)}

BUDGET = 50_000.0
REGION_MIN = 5_000.0
REGION_MAX = 12_000.0
H_MIN = 12_000.0
GAMMA = 0.002
LAM = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# Solver helpers
# ══════════════════════════════════════════════════════════════════════════════
def _status_ok(status):
    if status is None:
        return False
    s = str(status).lower()
    return ("optimal" in s) or (s == "1")


def _format_status(status):
    if status is None:
        return "Không chạy"
    s = str(status)
    if s.lower() in ["optimal", "optimal_inaccurate"] or s == "1":
        return "Optimal"
    if "infeasible" in s.lower() or s == "-1":
        return "Infeasible"
    if "unbounded" in s.lower() or s == "-2":
        return "Unbounded"
    return s


def _pack_result(method, status, z=None, x=None, message=""):
    if x is not None:
        x = np.array(x, dtype=float).reshape(6, 4)
        x[np.abs(x) < 1e-6] = 0.0
    return {
        "method": method,
        "status": _format_status(status),
        "raw_status": status,
        "Z": None if z is None else float(z),
        "A": x,
        "message": message,
    }


@st.cache_data(show_spinner=False)
def _solve_scipy(with_c5=True, lam=LAM, gamma=GAMMA, region_max=REGION_MAX):
    """Fallback LP solver bằng scipy.optimize.linprog/HiGHS."""
    if not HAS_SCIPY:
        return _pack_result("SciPy/HiGHS", None, message="Chưa cài scipy")

    n_x = 24
    has_M = bool(with_c5)
    n = n_x + (1 if has_M else 0)
    idx_M = n_x

    c = np.zeros(n)
    c[:n_x] = -BETA_TABLE.reshape(-1)  # maximize Z = minimize -Z

    A_ub = []
    b_ub = []

    # C1: tổng ngân sách <= 50.000
    row = np.zeros(n)
    row[:n_x] = 1.0
    A_ub.append(row)
    b_ub.append(BUDGET)

    # C2, C3: sàn/trần mỗi vùng
    for r in range(6):
        sl = slice(r * 4, (r + 1) * 4)
        row = np.zeros(n)
        row[sl] = -1.0
        A_ub.append(row)
        b_ub.append(-REGION_MIN)

        row = np.zeros(n)
        row[sl] = 1.0
        A_ub.append(row)
        b_ub.append(region_max)

    # C4: tổng H >= 12.000
    row = np.zeros(n)
    row[3:n_x:4] = -1.0
    A_ub.append(row)
    b_ub.append(-H_MIN)

    # C5: D_r + gamma*x_D,r >= lam * max_r(D_r + gamma*x_D,r)
    # Linear hóa: đặt M >= D_eff,r và D_eff,r >= lam*M
    if has_M:
        for r in range(6):
            # D0_r + gamma*x_D,r <= M  -> gamma*x_D,r - M <= -D0_r
            row = np.zeros(n)
            row[r * 4 + 1] = gamma
            row[idx_M] = -1.0
            A_ub.append(row)
            b_ub.append(-D0_ARRAY[r])

        for r in range(6):
            # D0_r + gamma*x_D,r >= lam*M -> -gamma*x_D,r + lam*M <= D0_r
            row = np.zeros(n)
            row[r * 4 + 1] = -gamma
            row[idx_M] = lam
            A_ub.append(row)
            b_ub.append(D0_ARRAY[r])

    bounds = [(0, None)] * n
    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return _pack_result("SciPy/HiGHS", res.status, message=res.message)

    return _pack_result("SciPy/HiGHS", "optimal", z=-res.fun, x=res.x[:n_x])


@st.cache_data(show_spinner=False)
def _solve_pulp(with_c5=True, lam=LAM, gamma=GAMMA, region_max=REGION_MAX):
    """Cài đặt đúng yêu cầu 4.4.1 bằng PuLP/CBC."""
    if not HAS_PULP:
        return _pack_result("PuLP/CBC", None, message="Môi trường hiện tại chưa cài PuLP")

    model = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (REGIONS, ITEMS), lowBound=0)

    model += pulp.lpSum(BETA[(r, j)] * x[r][j] for r in REGIONS for j in ITEMS), "GDP_gain"

    # C1
    model += pulp.lpSum(x[r][j] for r in REGIONS for j in ITEMS) <= BUDGET, "C1_Total_budget"

    # C2, C3
    for r in REGIONS:
        model += pulp.lpSum(x[r][j] for j in ITEMS) >= REGION_MIN, f"C2_Min_{r}"
        model += pulp.lpSum(x[r][j] for j in ITEMS) <= region_max, f"C3_Max_{r}"

    # C4
    model += pulp.lpSum(x[r]["H"] for r in REGIONS) >= H_MIN, "C4_Min_H"

    # C5
    if with_c5:
        M = pulp.LpVariable("Dmax", lowBound=0)
        for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] <= M, f"C5a_Dmax_{r}"
        for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] >= lam * M, f"C5b_Fair_{r}"

    try:
        status_code = model.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus.get(status_code, str(status_code))
    except Exception as exc:
        return _pack_result("PuLP/CBC", "error", message=str(exc))

    if status != "Optimal":
        return _pack_result("PuLP/CBC", status, message=f"PuLP status: {status}")

    A = np.array([[pulp.value(x[r][j]) for j in ITEMS] for r in REGIONS], dtype=float)
    return _pack_result("PuLP/CBC", status, z=pulp.value(model.objective), x=A)


@st.cache_data(show_spinner=False)
def _solve_cvxpy(with_c5=True, lam=LAM, gamma=GAMMA, region_max=REGION_MAX):
    """Cài đặt đúng yêu cầu 4.4.2 bằng CVXPY."""
    if not HAS_CVXPY:
        return _pack_result("CVXPY", None, message="Môi trường hiện tại chưa cài CVXPY")

    X = cp.Variable((6, 4), nonneg=True)
    constraints = []

    # C1
    constraints.append(cp.sum(X) <= BUDGET)

    # C2, C3
    for r in range(6):
        constraints.append(cp.sum(X[r, :]) >= REGION_MIN)
        constraints.append(cp.sum(X[r, :]) <= region_max)

    # C4
    constraints.append(cp.sum(X[:, 3]) >= H_MIN)

    # C5
    if with_c5:
        M = cp.Variable(nonneg=True)
        D_eff = D0_ARRAY + gamma * X[:, 1]
        constraints.append(D_eff <= M)
        constraints.append(D_eff >= lam * M)

    objective = cp.Maximize(cp.sum(cp.multiply(BETA_TABLE, X)))
    prob = cp.Problem(objective, constraints)

    installed = set(cp.installed_solvers())
    preferred = [s for s in ["CLARABEL", "ECOS", "SCS", "SCIPY"] if s in installed]

    last_error = ""
    for solver in preferred:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                A = np.array(X.value, dtype=float)
                return _pack_result(f"CVXPY/{solver}", prob.status, z=prob.value, x=A)
            if prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
                return _pack_result(f"CVXPY/{solver}", prob.status, message=f"CVXPY status: {prob.status}")
        except Exception as exc:
            last_error = str(exc)

    return _pack_result("CVXPY", prob.status, message=last_error or f"CVXPY status: {prob.status}")


def _choose_result(*results):
    """Ưu tiên PuLP, rồi CVXPY, rồi SciPy nếu có nghiệm tối ưu."""
    for res in results:
        if res["status"] == "Optimal" and res["A"] is not None:
            return res
    return results[0]


def _allocation_df(A):
    df = pd.DataFrame(A, columns=INAMES)
    df.insert(0, "Vùng", RNAMES)
    df["Tổng vùng"] = A.sum(axis=1)
    return df


def _solver_compare_df(results):
    rows = []
    for res in results:
        rows.append({
            "Phương pháp": res["method"],
            "Trạng thái": res["status"],
            "Z*": None if res["Z"] is None else round(res["Z"], 4),
            "Ghi chú": res["message"],
        })
    return pd.DataFrame(rows)


def _plot_heatmap(A, title="Heatmap phân bổ ngân sách"):
    fig = go.Figure(go.Heatmap(
        z=A,
        x=["I", "D", "AI", "H"],
        y=RNAMES,
        colorscale=[[0, "#F5F9F6"], [0.3, "#A5D6A7"], [0.6, "#4CAF72"], [1, "#1A6B3C"]],
        text=A,
        texttemplate="%{text:,.0f}",
        textfont=dict(size=11),
        colorbar=dict(title="tỷ VND", thickness=12, len=0.85),
        hovertemplate="%{y} · %{x}<br>%{z:,.0f} tỷ VND<extra></extra>",
    ))
    fig.update_layout(
        height=330,
        title=dict(text=title, font=dict(size=14), x=0.02),
        margin=dict(l=0, r=0, t=38, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CF,
        xaxis=dict(tickfont=dict(size=12), side="bottom"),
        yaxis=dict(tickfont=dict(size=10.5)),
    )
    return fig


def _plot_region_bar(A, title="Tổng ngân sách theo vùng"):
    totals = A.sum(axis=1)
    order = np.argsort(-totals)
    fig = go.Figure(go.Bar(
        x=[totals[i] for i in order],
        y=[RNAMES[i] for i in order],
        orientation="h",
        marker_color=[C1 if totals[i] >= 11_000 else C3 for i in order],
        marker_line_width=0,
        text=[f"{totals[i]:,.0f}" for i in order],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        height=300,
        title=dict(text=title, font=dict(size=14), x=0.02),
        margin=dict(l=0, r=45, t=38, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=CF,
        showlegend=False,
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        xaxis=dict(showgrid=True, gridcolor="#F0F4F0", tickfont=dict(size=10), range=[0, 14_000]),
    )
    return fig


def _explain_infeasible():
    max_initial = float(D0_ARRAY.max())  # Đông Nam Bộ = 82
    weakest_region_idx = int(np.argmin(D0_ARRAY))  # Tây Nguyên = 32
    weakest_name = RNAMES[weakest_region_idx]
    weakest_d0 = float(D0_ARRAY[weakest_region_idx])
    weakest_max_eff = weakest_d0 + GAMMA * REGION_MAX
    required_min = LAM * max_initial
    gap = required_min - weakest_max_eff
    lam_max = weakest_max_eff / max_initial
    gamma_min = (required_min - weakest_d0) / REGION_MAX
    cap_min = (required_min - weakest_d0) / GAMMA
    return {
        "max_initial": max_initial,
        "weakest_name": weakest_name,
        "weakest_d0": weakest_d0,
        "weakest_max_eff": weakest_max_eff,
        "required_min": required_min,
        "gap": gap,
        "lam_max": lam_max,
        "gamma_min": gamma_min,
        "cap_min": cap_min,
    }


def render():
    bai_header(
        so="4",
        ten="Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng",
        mo_ta="24 biến (6 vùng × 4 hạng mục), kiểm tra tính khả thi của C5, giải bằng PuLP/CVXPY và fallback SciPy",
        cap_do="TRUNG BÌNH",
        tools=["pulp", "cvxpy", "scipy", "plotly"],
        thoi_luong="1.5 tuần",
    )

    info_box(
        "Hàm mục tiêu: <b>max Z = ΣᵣΣⱼ βⱼ,ᵣ · xⱼ,ᵣ</b>. "
        "Ngân sách 50.000 tỷ VND cho 6 vùng × 4 hạng mục I, D, AI, H.<br>"
        "C5 được linear hóa bằng biến phụ <b>M</b>: "
        "Dᵣ + γx<sub>D,r</sub> ≤ M và Dᵣ + γx<sub>D,r</sub> ≥ λM.",
        bg="#E8F5E9", border=C1, icon="📐"
    )

    # ───────────────────────────────────────────────────────────────────────
    # Bảng dữ liệu
    # ───────────────────────────────────────────────────────────────────────
    section_title("Dữ liệu đầu vào của bài toán", "📊")
    df_beta = pd.DataFrame(BETA_TABLE, columns=INAMES)
    df_beta.insert(0, "Vùng", RNAMES)
    df_beta["D₀ — chỉ số số hóa ban đầu"] = D0_ARRAY.astype(int)
    st.dataframe(df_beta, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # Câu 4.4.1 & 4.4.2 — đúng đề, có C5
    # ════════════════════════════════════════════════════════════════════════
    section_title("Câu 4.4.1 & 4.4.2 — Giải đúng đề với ràng buộc công bằng C5", "🎯")

    pulp_c5 = _solve_pulp(with_c5=True, lam=LAM, gamma=GAMMA, region_max=REGION_MAX)
    cvx_c5 = _solve_cvxpy(with_c5=True, lam=LAM, gamma=GAMMA, region_max=REGION_MAX)
    scipy_c5 = _solve_scipy(with_c5=True, lam=LAM, gamma=GAMMA, region_max=REGION_MAX)

    st.dataframe(_solver_compare_df([pulp_c5, cvx_c5, scipy_c5]), use_container_width=True, hide_index=True)

    infea = _explain_infeasible()
    if scipy_c5["status"] != "Optimal":
        info_box(
            f"<b>Kết luận quan trọng:</b> Với đúng tham số đề bài λ={LAM}, γ={GAMMA}, "
            f"trần mỗi vùng={REGION_MAX:,.0f}, mô hình <b>có C5 là vô nghiệm</b>.<br><br>"
            f"Lý do: vùng có D₀ cao nhất là Đông Nam Bộ = {infea['max_initial']:.0f}, "
            f"nên ngưỡng tối thiểu theo C5 là λ·M ≥ {LAM}×{infea['max_initial']:.0f} "
            f"= <b>{infea['required_min']:.1f}</b>. "
            f"Trong khi {infea['weakest_name']} có D₀={infea['weakest_d0']:.0f}; ngay cả khi dồn toàn bộ "
            f"{REGION_MAX:,.0f} tỷ vào D thì D hiệu dụng tối đa chỉ là "
            f"{infea['weakest_d0']:.0f}+{GAMMA}×{REGION_MAX:,.0f}="
            f"<b>{infea['weakest_max_eff']:.1f}</b> &lt; {infea['required_min']:.1f}.",
            bg="#FCE4EC", border=CRED, icon="⚠️"
        )

        df_fix = pd.DataFrame([
            {"Cách nới điều kiện": "Giảm λ", "Ngưỡng tối thiểu": f"λ ≤ {infea['lam_max']:.4f}"},
            {"Cách nới điều kiện": "Tăng γ", "Ngưỡng tối thiểu": f"γ ≥ {infea['gamma_min']:.6f}"},
            {"Cách nới điều kiện": "Tăng trần vùng Tây Nguyên", "Ngưỡng tối thiểu": f"C3 ≥ {infea['cap_min']:,.0f} tỷ"},
        ])
        st.dataframe(df_fix, use_container_width=True, hide_index=True)
    else:
        best_c5 = _choose_result(pulp_c5, cvx_c5, scipy_c5)
        st.dataframe(_allocation_df(best_c5["A"]), use_container_width=True, hide_index=True)
        info_box(
            f"Z* có C5 = <b>{best_c5['Z']:,.2f}</b> tỷ VND GDP gain. "
            "PuLP/CBC và CVXPY cho kết quả tương đương, sai khác chỉ do dung sai solver.",
            bg="#E3F2FD", border=CBLUE, icon="✅"
        )

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # Câu 4.4.4 — bỏ C5
    # ════════════════════════════════════════════════════════════════════════
    section_title("Câu 4.4.4 — Mô hình bỏ ràng buộc công bằng C5", "⚖️")

    pulp_no = _solve_pulp(with_c5=False)
    cvx_no = _solve_cvxpy(with_c5=False)
    scipy_no = _solve_scipy(with_c5=False)
    st.dataframe(_solver_compare_df([pulp_no, cvx_no, scipy_no]), use_container_width=True, hide_index=True)

    best_no = _choose_result(pulp_no, cvx_no, scipy_no)
    if best_no["status"] == "Optimal":
        A_no = best_no["A"]
        z_no = best_no["Z"]

        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.metric("Z* bỏ C5", f"{z_no:,.0f}", "tỷ VND GDP gain")
        with col2:
            st.metric("Ngân sách dùng", f"{A_no.sum():,.0f}", "/ 50.000")
        with col3:
            st.metric("Tổng H", f"{A_no[:, 3].sum():,.0f}", "≥ 12.000")

        st.dataframe(_allocation_df(A_no), use_container_width=True, hide_index=True)

        col_hm, col_bar = st.columns([3, 2], gap="large")
        with col_hm:
            st.plotly_chart(_plot_heatmap(A_no, "Heatmap — nghiệm tối ưu khi bỏ C5"), use_container_width=True, config={"displayModeBar": False})
        with col_bar:
            st.plotly_chart(_plot_region_bar(A_no, "Tổng ngân sách bỏ C5"), use_container_width=True, config={"displayModeBar": False})

        info_box(
            "<b>Nhận xét:</b> Khi bỏ C5, ngân sách dồn vào các hạng mục có hệ số biên cao nhất: "
            "AI ở ĐB sông Hồng và Đông Nam Bộ, H ở Tây Nguyên, Trung du MN Bắc, Bắc TB + DHMT và ĐB sông Cửu Long. "
            "Tây Nguyên không chỉ nhận mức sàn 5.000 tỷ mà nhận 11.000 tỷ vào H vì β<sub>H,CH</sub>=1,35 là rất cao.",
            bg="#E8F5E9", border=C3, icon="💡"
        )
    else:
        info_box("Mô hình bỏ C5 cũng không giải được trong môi trường hiện tại.", bg="#FCE4EC", border=CRED, icon="⚠️")

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # Kịch bản tham khảo để vẫn trình bày được heatmap công bằng
    # ════════════════════════════════════════════════════════════════════════
    section_title("Kịch bản tham khảo: nới nhẹ λ để C5 khả thi", "🧪")

    lam_adj = 0.68
    pulp_adj = _solve_pulp(with_c5=True, lam=lam_adj, gamma=GAMMA, region_max=REGION_MAX)
    cvx_adj = _solve_cvxpy(with_c5=True, lam=lam_adj, gamma=GAMMA, region_max=REGION_MAX)
    scipy_adj = _solve_scipy(with_c5=True, lam=lam_adj, gamma=GAMMA, region_max=REGION_MAX)
    best_adj = _choose_result(pulp_adj, cvx_adj, scipy_adj)

    info_box(
        f"Phần này <b>không thay thế đáp án đúng đề</b>. Nó chỉ minh họa nếu giảng viên muốn một nghiệm có công bằng để vẽ heatmap. "
        f"Ta nới λ từ 0.70 xuống <b>{lam_adj:.2f}</b>, nhỏ hơn ngưỡng khả thi tối đa {infea['lam_max']:.4f}.",
        bg="#FFF8E1", border=CORANGE, icon="ℹ️"
    )
    st.dataframe(_solver_compare_df([pulp_adj, cvx_adj, scipy_adj]), use_container_width=True, hide_index=True)

    if best_adj["status"] == "Optimal" and best_no["status"] == "Optimal":
        A_adj = best_adj["A"]
        z_adj = best_adj["Z"]
        cost = z_no - z_adj
        cost_pct = cost / z_no * 100 if z_no else np.nan

        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.metric("Z* có C5, λ=0.68", f"{z_adj:,.0f}", "tỷ VND GDP gain")
        with col2:
            st.metric("Chi phí công bằng", f"{cost:,.0f}", "so với bỏ C5")
        with col3:
            st.metric("Tổn thất tương đối", f"{cost_pct:.1f}%", "GDP gain")

        st.dataframe(_allocation_df(A_adj), use_container_width=True, hide_index=True)

        D_eff = D0_ARRAY + GAMMA * A_adj[:, 1]
        df_deff = pd.DataFrame({
            "Vùng": RNAMES,
            "D₀": D0_ARRAY,
            "x_D": A_adj[:, 1],
            "D hiệu dụng = D₀ + γx_D": np.round(D_eff, 2),
        })
        st.dataframe(df_deff, use_container_width=True, hide_index=True)

        col_hm, col_bar = st.columns([3, 2], gap="large")
        with col_hm:
            st.plotly_chart(_plot_heatmap(A_adj, "Heatmap — kịch bản công bằng khả thi λ=0.68"), use_container_width=True, config={"displayModeBar": False})
        with col_bar:
            st.plotly_chart(_plot_region_bar(A_adj, "Tổng ngân sách λ=0.68"), use_container_width=True, config={"displayModeBar": False})

        # So sánh tổng vùng bỏ C5 vs λ=0.68
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="Bỏ C5", x=RNAMES, y=A_no.sum(axis=1), marker_color=CORANGE, marker_line_width=0))
        fig_cmp.add_trace(go.Bar(name="C5 λ=0.68", x=RNAMES, y=A_adj.sum(axis=1), marker_color=C1, marker_line_width=0))
        fig_cmp.update_layout(
            height=300,
            title=dict(text="So sánh tổng ngân sách: bỏ C5 vs C5 khả thi λ=0.68", font=dict(size=14), x=0.02),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=CF,
            barmode="group",
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F0", tickfont=dict(size=10)),
            xaxis=dict(showgrid=False, tickangle=-20, tickfont=dict(size=9.5)),
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

        info_box(
            f"Trong kịch bản λ=0.68, chi phí công bằng = <b>{cost:,.0f}</b> tỷ VND GDP gain "
            f"({cost_pct:.1f}%). Vùng yếu số như Tây Nguyên, Trung du MN Bắc và ĐB sông Cửu Long "
            "phải nhận thêm ngân sách D để kéo D hiệu dụng lên sát ngưỡng công bằng.",
            bg="#FFF8E1", border=CORANGE, icon="💡"
        )
    else:
        info_box("Không tạo được nghiệm minh họa λ=0.68 trong môi trường hiện tại.", bg="#FCE4EC", border=CRED, icon="⚠️")

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # Câu hỏi thảo luận chính sách
    # ════════════════════════════════════════════════════════════════════════
    section_title("Câu hỏi thảo luận chính sách", "💬")

    with st.expander("a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào? Tại sao?", expanded=False):
        info_box(
            "Nếu bỏ C5, vốn chảy mạnh vào <b>ĐB sông Hồng</b> và <b>Đông Nam Bộ</b> cho hạng mục AI "
            "vì hệ số AI lần lượt là 1,40 và 1,55. Ngoài ra, mô hình vẫn phân bổ đáng kể vào "
            "<b>H tại Tây Nguyên</b> do hệ số H của vùng này là 1,35. Hậu quả dài hạn nếu bỏ hẳn công bằng: "
            "khoảng cách số giữa vùng lõi và vùng yếu có thể tăng, tạo rủi ro bất bình đẳng vùng miền, di cư lao động "
            "và giảm tính bao trùm của chiến lược kinh tế số.",
            bg="#F1F8F2", border=C2, icon="✅"
        )

    with st.expander("b) Ràng buộc trần ngân sách mỗi vùng C3 làm giảm Z* bao nhiêu phần trăm?", expanded=False):
        info_box(
            "Trong phạm vi bài này, có thể xem C3 là chính sách chống tập trung vốn quá mức. Muốn tính riêng tác động của C3, "
            "cần chạy thêm một mô hình bỏ trần vùng và so với nghiệm có trần. Tuy nhiên nếu giữ đúng C5 với λ=0.7 thì mô hình "
            "đang vô nghiệm, nên không thể tính phần trăm giảm Z* của C3 trong cấu hình gốc. Về chính sách, C3 có thể làm giảm "
            "hiệu quả ngắn hạn nhưng giúp phân quyền, giảm tập trung và tăng tính bền vững xã hội.",
            bg="#F1F8F2", border=C2, icon="✅"
        )

    with st.expander("c) Tây Nguyên nên đầu tư AI hay H/I trước?", expanded=False):
        info_box(
            "Mô hình trả lời khá rõ: <b>không nên ưu tiên AI tại Tây Nguyên ngay từ đầu</b>, vì β_AI chỉ 0,45, thấp nhất trong các hạng mục. "
            "Nếu bỏ C5, mô hình dồn 11.000 tỷ vào H tại Tây Nguyên. Nếu áp công bằng khả thi, Tây Nguyên phải đầu tư mạnh vào D để kéo chỉ số số hóa lên. "
            "Hàm ý chính sách: Tây Nguyên nên ưu tiên <b>nhân lực số H</b> và <b>chuyển đổi số nền tảng D</b> trước, sau đó mới mở rộng AI khi năng lực hấp thụ tốt hơn.",
            bg="#F1F8F2", border=C2, icon="✅"
        )

    end_padding()
