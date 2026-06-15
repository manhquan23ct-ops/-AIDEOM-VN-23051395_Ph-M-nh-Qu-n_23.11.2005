"""pages/bai7.py — Bài 7: Tối ưu đa mục tiêu Pareto với NSGA-II

Bản làm lại:
- Cài đặt NSGA-II đúng yêu cầu pop_size=100, n_gen=200.
- Bổ sung kiểm tra tính khả thi của ràng buộc công bằng C5 từ Bài 4.
- Với tham số gốc λ=0.70, γ=0.002, trần vùng 12.000, mô hình vô nghiệm.
  Vì vậy phần NSGA-II minh họa dùng λ=0.68 để có tập Pareto khả thi, đồng thời cảnh báo rõ trong giao diện.
- Cài đặt đầy đủ 24 biến, 4 mục tiêu, TOPSIS chọn nghiệm thỏa hiệp, chi phí cơ hội.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import bai_header, end_padding, info_box, section_title

# ── Style ───────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"; C2 = "#2E8B57"; C3 = "#4CAF72"
CBLUE = "#1976D2"; CRED = "#E53935"; CORANGE = "#E65100"; CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

# ── Dữ liệu Bài 4 + tham số Bài 7 ────────────────────────────────────────────
REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_VI = ["TDMNPB", "ĐBSH", "BTB+DHMT", "Tây Nguyên", "ĐNB", "ĐBSCL"]
REGION_FULL = [
    "Trung du miền núi phía Bắc", "Đồng bằng sông Hồng", "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên", "Đông Nam Bộ", "Đồng bằng sông Cửu Long"
]
ITEMS = ["I", "D", "AI", "H"]
ITEM_VI = ["Hạ tầng số", "CĐS DN", "AI", "Nhân lực"]

# Ma trận beta: hàng = vùng, cột = I, D, AI, H
BETA = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
], dtype=float)

# Chỉ số số hóa ban đầu D0 từ Bài 4
D0 = np.array([38, 78, 55, 32, 82, 48], dtype=float)
GAMMA = 0.002
LAM_ORIGINAL = 0.70
LAM_RUN = 0.68      # nới nhẹ để mô hình khả thi; xem cảnh báo trong app

# Tham số Bài 7
E = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38], dtype=float)
RHO = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22], dtype=float)
SIG = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30], dtype=float)

BUDGET = 50000.0
FLOOR = 5000.0
CEIL = 12000.0
H_FLOOR = 12000.0

POLICY_W = np.array([0.40, 0.25, 0.20, 0.15], dtype=float)
IS_BENEFIT = np.array([True, False, False, False])  # GDP là lợi ích, 3 mục tiêu còn lại là chi phí

# ─────────────────────────────────────────────────────────────────────────────
# Hàm kiểm tra mô hình và tính mục tiêu
# ─────────────────────────────────────────────────────────────────────────────

def original_c5_feasibility_bound():
    """Trả về các đại lượng chứng minh C5 gốc λ=0.70 là vô nghiệm."""
    min_M = float(D0.max())
    required_threshold = LAM_ORIGINAL * min_M
    ch_max_eff = float(D0[3] + GAMMA * CEIL)  # Tây Nguyên nếu dồn toàn bộ trần vùng vào D
    lam_max_feasible = ch_max_eff / min_M
    gamma_min = (required_threshold - D0[3]) / CEIL
    return min_M, required_threshold, ch_max_eff, lam_max_feasible, gamma_min


def objectives_from_X(X):
    """Trả về F_display = [GDP gain, MAD bất bình đẳng, CO2, rủi ro ròng]."""
    gdp = float((BETA * X).sum())
    region_budget = X.sum(axis=1)
    mad = float(np.abs(region_budget - region_budget.mean()).mean())
    co2 = float((E * (X[:, 0] + X[:, 2])).sum())
    risk = float((RHO * X[:, 2]).sum() - (SIG * X[:, 3]).sum())
    return np.array([gdp, mad, co2, risk], dtype=float)


def constraint_violations(X, lam=LAM_RUN):
    """Trả về vector g(x) <= 0 cho các ràng buộc C1-C5.

    C6 không âm được xử lý bằng bounds xl=0.
    Số ràng buộc: 1 + 6 + 6 + 1 + 6 = 20.
    """
    region_sum = X.sum(axis=1)
    eff_digital = D0 + GAMMA * X[:, 1]
    M_eff = eff_digital.max()

    G = []
    G.append(X.sum() - BUDGET)                 # C1: ngân sách tổng <= 50.000
    G.extend(FLOOR - region_sum)               # C2: mỗi vùng >= 5.000
    G.extend(region_sum - CEIL)                # C3: mỗi vùng <= 12.000
    G.append(H_FLOOR - X[:, 3].sum())          # C4: tổng H >= 12.000
    G.extend(lam * M_eff - eff_digital)        # C5: D_r + γx_D,r >= λM
    return np.array(G, dtype=float)


def is_feasible(X, lam=LAM_RUN, tol=1e-6):
    return bool(np.all(constraint_violations(X, lam) <= tol) and np.all(X >= -tol))


def topsis(F, weights, is_benefit):
    """TOPSIS trên ma trận F_display.

    F[:,0] càng lớn càng tốt; F[:,1:4] càng nhỏ càng tốt.
    """
    X = F.astype(float)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1e-12
    R = X / norm
    V = R * w
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    C = S_neg / (S_star + S_neg + 1e-12)
    return C, S_star, S_neg


def non_dominated_mask(F):
    """Mask nghiệm không bị trội. F ở dạng minimization: [-GDP, MAD, CO2, risk]."""
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by_any = np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
        if dominated_by_any:
            keep[i] = False
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Fallback khả thi nếu máy chưa có pymoo
# ─────────────────────────────────────────────────────────────────────────────

def _base_feasible_X(lam=LAM_RUN):
    """Tạo một nghiệm khả thi nền theo logic C5 với M = max D0 = 82.

    Tránh tăng x_D ở ĐBSH và ĐNB để không làm M tăng, từ đó giữ C5 dễ thỏa.
    """
    X = np.zeros((6, 4), dtype=float)
    target = lam * D0.max()

    # Đầu tư D tối thiểu cho vùng có D0 thấp để đạt target
    for r in range(6):
        req_d = max(0.0, (target - D0[r]) / GAMMA)
        if r in [1, 4]:  # ĐBSH, ĐNB đã là vùng max, không cần tăng D trong nghiệm nền
            req_d = 0.0
        X[r, 1] = min(req_d, CEIL)

    # Đảm bảo sàn mỗi vùng bằng H vì H giảm rủi ro và có ích chính sách
    for r in range(6):
        gap = FLOOR - X[r].sum()
        if gap > 0:
            X[r, 3] += min(gap, CEIL - X[r].sum())

    # Đảm bảo sàn nhân lực số
    h_gap = H_FLOOR - X[:, 3].sum()
    if h_gap > 0:
        order = np.argsort(-SIG)  # vùng H giảm rủi ro tốt hơn được ưu tiên
        for r in order:
            add = min(h_gap, CEIL - X[r].sum())
            if add > 0:
                X[r, 3] += add
                h_gap -= add
            if h_gap <= 1e-9:
                break
    return X


def random_feasible_population(n=120, seed=42, lam=LAM_RUN):
    rng = np.random.default_rng(seed)
    base = _base_feasible_X(lam)
    cells = []
    for r in range(6):
        for j in range(4):
            # Không tăng D ở ĐBSH/ĐNB để tránh làm M tăng và kéo C5 khó hơn.
            if j == 1 and r in [1, 4]:
                continue
            cells.append((r, j))

    pop = []
    for _ in range(n):
        X = base.copy()
        remaining = BUDGET - X.sum()
        # Dùng một phần ngẫu nhiên ngân sách còn lại để tạo đa dạng Pareto
        use = remaining * rng.uniform(0.25, 1.0)
        for _step in range(400):
            if use <= 1e-6:
                break
            r, j = cells[rng.integers(0, len(cells))]
            cap = CEIL - X[r].sum()
            if cap <= 1e-9:
                continue
            add = min(cap, use, rng.exponential(900.0))
            X[r, j] += add
            use -= add

        if is_feasible(X, lam):
            pop.append(X.reshape(-1))

    if not pop:
        pop = [base.reshape(-1)]
    return np.array(pop, dtype=float)


@st.cache_data(show_spinner=False)
def fallback_pareto(n=300, seed=42, lam=LAM_RUN):
    Xs = random_feasible_population(n, seed, lam)
    F_display = np.array([objectives_from_X(x.reshape(6, 4)) for x in Xs])
    F_min = F_display.copy()
    F_min[:, 0] = -F_min[:, 0]
    mask = non_dominated_mask(F_min)
    return F_display[mask], Xs[mask]


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-II bằng pymoo
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_nsga(pop_size=100, n_gen=200, seed=42, lam=LAM_RUN):
    """Chạy NSGA-II. Trả về (F_display, X_pareto, source)."""
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.core.sampling import Sampling
    except Exception:
        F_fb, X_fb = fallback_pareto(500, seed, lam)
        return F_fb, X_fb, "fallback"

    class FeasibleSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            return random_feasible_population(n_samples, seed, lam)

    class VietnamDigitalProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=20,
                xl=np.zeros(24),
                xu=np.ones(24) * CEIL,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            X = x.reshape(6, 4)
            F_disp = objectives_from_X(X)
            # pymoo mặc định minimize: max GDP => minimize -GDP
            out["F"] = np.array([-F_disp[0], F_disp[1], F_disp[2], F_disp[3]], dtype=float)
            out["G"] = constraint_violations(X, lam)

    problem = VietnamDigitalProblem()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FeasibleSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=int(seed), verbose=False)

    if res.F is not None and len(res.F) > 0:
        F_min = np.array(res.F, dtype=float)
        X = np.array(res.X, dtype=float)
    else:
        feas = [ind for ind in res.pop if ind.feas]
        if not feas:
            F_fb, X_fb = fallback_pareto(500, seed, lam)
            return F_fb, X_fb, "fallback"
        F_min = np.array([ind.F for ind in feas], dtype=float)
        X = np.array([ind.X for ind in feas], dtype=float)

    F_display = F_min.copy()
    F_display[:, 0] = -F_display[:, 0]
    return F_display, X, "pymoo"


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    bai_header(
        so="7",
        ten="Tối ưu đa mục tiêu Pareto với NSGA-II",
        mo_ta="24 biến, 4 mục tiêu xung đột: tăng trưởng, bao trùm, môi trường, an ninh dữ liệu",
        cap_do="KHÁ KHÓ",
        tools=["pymoo", "NSGA-II", "TOPSIS", "plotly"],
        thoi_luong="2 tuần",
    )

    min_M, req, ch_max, lam_max, gamma_min = original_c5_feasibility_bound()
    info_box(
        "Mô hình gốc Bài 7 kế thừa ràng buộc C5 từ Bài 4. Với λ=0,70 và γ=0,002, "
        "mô hình <b>không khả thi</b>: Tây Nguyên tối đa chỉ đạt "
        f"D hiệu dụng = 32 + 0,002×12.000 = <b>{ch_max:.1f}</b>, trong khi ngưỡng tối thiểu là "
        f"0,70×82 = <b>{req:.1f}</b>. Vì vậy bản này chạy NSGA-II minh họa với "
        f"<b>λ = {LAM_RUN:.2f}</b> để có tập Pareto khả thi, đồng thời giữ nguyên các ràng buộc còn lại.",
        bg="#FFF8E1", border=CORANGE, icon="⚠️"
    )

    info_box(
        "4 mục tiêu: <b>max f₁ GDP gain</b>; <b>min f₂ bất bình đẳng vùng</b> "
        "(MAD ngân sách vùng); <b>min f₃ phát thải CO₂</b>; <b>min f₄ rủi ro an ninh dữ liệu ròng</b>.<br>"
        "Ràng buộc: C1 ngân sách tổng, C2 sàn vùng, C3 trần vùng, C4 sàn nhân lực, C5 công bằng số, C6 không âm.",
        bg="#E8F5E9", border=C1, icon="📐"
    )

    # Tham số mô hình
    section_title("Mô hình và tham số", "📋")
    tab_beta, tab_params, tab_c5 = st.tabs(["β tăng trưởng", "e, ρ, σ", "Kiểm tra C5"])
    with tab_beta:
        df_beta = pd.DataFrame(BETA, columns=ITEM_VI)
        df_beta.insert(0, "Vùng", REGION_FULL)
        st.dataframe(df_beta, use_container_width=True, hide_index=True)
    with tab_params:
        df_p = pd.DataFrame({
            "Vùng": REGION_FULL,
            "eᵣ CO₂/tỷ": E,
            "ρᵣ rủi ro/AI": RHO,
            "σᵣ giảm rủi ro/H": SIG,
            "D₀": D0,
        })
        st.dataframe(df_p, use_container_width=True, hide_index=True)
    with tab_c5:
        df_c5 = pd.DataFrame({
            "Chỉ tiêu": [
                "M tối thiểu do D₀ cao nhất", "Ngưỡng C5 với λ=0,70", "Tây Nguyên tối đa khi x_D=12.000",
                "λ tối đa để còn khả thi", "γ tối thiểu nếu giữ λ=0,70"
            ],
            "Giá trị": [
                f"{min_M:.1f}", f"{req:.1f}", f"{ch_max:.1f}", f"{lam_max:.4f}", f"{gamma_min:.6f}"
            ],
            "Ý nghĩa": [
                "Đông Nam Bộ có D₀ = 82", "Mọi vùng phải đạt ít nhất 57,4", "Tây Nguyên chỉ đạt 56,0 < 57,4",
                "λ phải ≤ mức này nếu giữ γ=0,002", "Hoặc tăng γ lên mức này nếu giữ λ=0,70"
            ]
        })
        st.dataframe(df_c5, use_container_width=True, hide_index=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── Câu 7.4.1 ────────────────────────────────────────────────────────────
    section_title("Câu 7.4.1 — Chạy NSGA-II", "🎯")
    with st.spinner("Đang chạy NSGA-II: pop_size=100, n_gen=200..."):
        F_disp, X_pareto, source = run_nsga(100, 200, 42, LAM_RUN)

    n_p = len(F_disp)
    source_text = "pymoo NSGA-II" if source == "pymoo" else "fallback non-dominated sampling"

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    kpis = [
        (k1, "Nguồn nghiệm", source_text, "solver", CPURPLE),
        (k2, "Số nghiệm Pareto", f"{n_p}", "phương án", C1),
        (k3, "GDP max", f"{F_disp[:,0].max():,.0f}", "tỷ VND", CBLUE),
        (k4, "Bất BĐ min", f"{F_disp[:,1].min():,.0f}", "MAD", C2),
        (k5, "CO₂ min", f"{F_disp[:,2].min():,.1f}", "đơn vị", CORANGE),
    ]
    for col, lbl, val, sub, color in kpis:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:0.9rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);text-align:center;height:100%;">
                <div style="font-size:0.72rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:5px;">{lbl}</div>
                <div style="font-size:1.15rem;font-weight:900;color:{color};line-height:1.15;">{val}</div>
                <div style="font-size:0.70rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── Câu 7.4.2 ────────────────────────────────────────────────────────────
    section_title("Câu 7.4.2 — Trích xuất Pareto và trực quan hóa", "🔵")

    tab3d, tabpc, tabtbl = st.tabs(["Scatter 3D", "Parallel coordinates", "Bảng Pareto"])
    with tab3d:
        fig3d = go.Figure(go.Scatter3d(
            x=F_disp[:, 0], y=F_disp[:, 1], z=F_disp[:, 2], mode="markers",
            marker=dict(
                size=4, color=F_disp[:, 3], colorscale="RdYlGn_r",
                colorbar=dict(title="f₄ rủi ro", len=0.75), opacity=0.82
            ),
            hovertemplate="GDP: %{x:,.0f}<br>BĐ: %{y:,.0f}<br>CO₂: %{z:,.1f}<br>Risk: %{marker.color:.1f}<extra></extra>",
        ))
        fig3d.update_layout(
            height=520, margin=dict(l=0, r=0, t=10, b=0), font=CF,
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(xaxis_title="f₁ GDP gain", yaxis_title="f₂ bất bình đẳng", zaxis_title="f₃ CO₂"),
        )
        st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})
        st.caption("Mỗi điểm là một phương án Pareto: muốn tăng một mục tiêu thường phải hy sinh ít nhất một mục tiêu khác.")

    with tabpc:
        fig_pc = go.Figure(go.Parcoords(
            line=dict(color=F_disp[:, 0], colorscale="Viridis", colorbar=dict(title="GDP", len=0.8)),
            dimensions=[
                dict(label="f₁ GDP", values=F_disp[:, 0]),
                dict(label="f₂ Bất BĐ", values=F_disp[:, 1]),
                dict(label="f₃ CO₂", values=F_disp[:, 2]),
                dict(label="f₄ Rủi ro", values=F_disp[:, 3]),
            ]
        ))
        fig_pc.update_layout(height=430, margin=dict(l=40, r=40, t=20, b=20), font=CF,
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pc, use_container_width=True, config={"displayModeBar": False})

    with tabtbl:
        show_n = min(20, n_p)
        df_pf = pd.DataFrame({
            "GDP gain": F_disp[:, 0],
            "Bất bình đẳng MAD": F_disp[:, 1],
            "CO₂": F_disp[:, 2],
            "Rủi ro ròng": F_disp[:, 3],
        }).sort_values("GDP gain", ascending=False).head(show_n)
        st.dataframe(df_pf.round(2), use_container_width=True, hide_index=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── Câu 7.4.3 ────────────────────────────────────────────────────────────
    section_title("Câu 7.4.3 — Chọn nghiệm thỏa hiệp bằng TOPSIS", "⭐")

    C_top, S_star, S_neg = topsis(F_disp, POLICY_W, IS_BENEFIT)
    best_idx = int(np.argmax(C_top))
    F_best = F_disp[best_idx]
    X_best = X_pareto[best_idx].reshape(6, 4)

    info_box(
        "TOPSIS dùng trọng số chính sách: <b>0,40 tăng trưởng</b>; <b>0,25 bao trùm</b>; "
        "<b>0,20 môi trường</b>; <b>0,15 an ninh dữ liệu</b>. GDP là tiêu chí lợi ích; 3 tiêu chí còn lại là chi phí.",
        bg="#E3F2FD", border=CBLUE, icon="⚖️"
    )

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    kpi_best = [
        (c1, "TOPSIS C*", f"{C_top[best_idx]:.4f}", "cao nhất", C1),
        (c2, "GDP", f"{F_best[0]:,.0f}", "tỷ VND", CBLUE),
        (c3, "Bất BĐ", f"{F_best[1]:,.0f}", "MAD", C2),
        (c4, "CO₂", f"{F_best[2]:,.1f}", "phát thải", CORANGE),
        (c5, "Rủi ro", f"{F_best[3]:,.1f}", "ròng", CRED),
    ]
    for col, lbl, val, sub, color in kpi_best:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:0.9rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);text-align:center;">
                <div style="font-size:0.72rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:5px;">{lbl}</div>
                <div style="font-size:1.35rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                <div style="font-size:0.70rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    col_alloc, col_mark = st.columns([2, 3], gap="large")
    with col_alloc:
        st.markdown("""
        <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                    padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
            <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:0.6rem;">
                Ma trận phân bổ nghiệm thỏa hiệp</div>
        """, unsafe_allow_html=True)
        df_best = pd.DataFrame(X_best, columns=ITEM_VI)
        df_best.insert(0, "Vùng", REGION_VI)
        df_best["Tổng vùng"] = X_best.sum(axis=1)
        st.dataframe(df_best.round(0), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_mark:
        fig_mark = go.Figure()
        fig_mark.add_trace(go.Scatter(
            x=F_disp[:, 0], y=F_disp[:, 1], mode="markers",
            marker=dict(size=7, color=F_disp[:, 2], colorscale="Blues", opacity=0.65,
                        colorbar=dict(title="CO₂", len=0.75)),
            name="Pareto",
            hovertemplate="GDP:%{x:,.0f}<br>BĐ:%{y:,.0f}<extra></extra>",
        ))
        fig_mark.add_trace(go.Scatter(
            x=[F_best[0]], y=[F_best[1]], mode="markers+text",
            marker=dict(size=20, color=CRED, symbol="star", line=dict(width=2, color="white")),
            text=["★"], textposition="middle center", name="Thỏa hiệp",
        ))
        fig_mark.update_layout(
            height=330, margin=dict(l=0, r=0, t=10, b=0), font=CF,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(title="f₁ GDP gain", showgrid=True, gridcolor="#F0F4F0"),
            yaxis=dict(title="f₂ bất bình đẳng", showgrid=True, gridcolor="#F0F4F0"),
        )
        st.plotly_chart(fig_mark, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── Câu 7.4.4 ────────────────────────────────────────────────────────────
    section_title("Câu 7.4.4 — Chi phí cơ hội của mục tiêu tăng trưởng", "💸")

    idx_gdp = int(np.argmax(F_disp[:, 0]))
    idx_ineq = int(np.argmin(F_disp[:, 1]))
    idx_co2 = int(np.argmin(F_disp[:, 2]))
    idx_risk = int(np.argmin(F_disp[:, 3]))
    F_gdp = F_disp[idx_gdp]

    scenarios = [
        ("★ Thỏa hiệp TOPSIS", F_best, C1),
        ("📈 GDP cao nhất", F_gdp, CBLUE),
        ("🤝 Bao trùm nhất", F_disp[idx_ineq], CPURPLE),
        ("🌿 Xanh nhất", F_disp[idx_co2], C2),
        ("🔒 An toàn nhất", F_disp[idx_risk], CRED),
    ]
    df_sc = pd.DataFrame([
        {"Kịch bản": name, "GDP gain": f[0], "Bất BĐ": f[1], "CO₂": f[2], "Rủi ro": f[3]}
        for name, f, _ in scenarios
    ])
    st.dataframe(df_sc.round(2), use_container_width=True, hide_index=True)

    d_gdp = (F_gdp[0] - F_best[0]) / max(abs(F_best[0]), 1) * 100
    d_ineq = (F_gdp[1] - F_best[1]) / max(abs(F_best[1]), 1) * 100
    d_co2 = (F_gdp[2] - F_best[2]) / max(abs(F_best[2]), 1) * 100
    d_risk = (F_gdp[3] - F_best[3]) / max(abs(F_best[3]), 1) * 100 if abs(F_best[3]) > 1e-9 else np.nan

    cc1, cc2, cc3, cc4 = st.columns(4, gap="medium")
    cost_cards = [
        (cc1, "GDP tăng thêm", f"+{d_gdp:.1f}%", "so với thỏa hiệp", C1),
        (cc2, "Bất bình đẳng", f"{d_ineq:+.1f}%", "tăng là xấu", CRED),
        (cc3, "CO₂", f"{d_co2:+.1f}%", "tăng là xấu", CORANGE),
        (cc4, "Rủi ro", "N/A" if np.isnan(d_risk) else f"{d_risk:+.1f}%", "tăng là xấu", CPURPLE),
    ]
    for col, lbl, val, sub, color in cost_cards:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);text-align:center;">
                <div style="font-size:0.76rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:5px;">{lbl}</div>
                <div style="font-size:1.45rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                <div style="font-size:0.74rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Radar chuẩn hóa các kịch bản
    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                padding:1.2rem 1.4rem 0.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
        <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:0.6rem;">
            Radar — So sánh 5 kịch bản, mọi trục càng cao càng tốt</div>
    """, unsafe_allow_html=True)
    dims = ["GDP", "Bao trùm", "Xanh", "An toàn"]
    fig_radar = go.Figure()
    for name, f, color in scenarios:
        vals = [
            (f[0] - F_disp[:, 0].min()) / (np.ptp(F_disp[:, 0]) + 1e-12),
            1 - (f[1] - F_disp[:, 1].min()) / (np.ptp(F_disp[:, 1]) + 1e-12),
            1 - (f[2] - F_disp[:, 2].min()) / (np.ptp(F_disp[:, 2]) + 1e-12),
            1 - (f[3] - F_disp[:, 3].min()) / (np.ptp(F_disp[:, 3]) + 1e-12),
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]], fill="toself",
            name=name, line_color=color, opacity=0.55
        ))
    fig_radar.update_layout(
        height=440, margin=dict(l=40, r=40, t=20, b=40), font=CF,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── Thảo luận chính sách ────────────────────────────────────────────────
    section_title("Câu hỏi thảo luận chính sách", "💬")
    corr_12 = np.corrcoef(F_disp[:, 0], F_disp[:, 1])[0, 1] if n_p > 2 else np.nan
    q_data = [
        (
            "a)",
            "Đánh đổi tăng trưởng — bao trùm trên đường biên Pareto có rõ không?",
            f"Tương quan thực nghiệm giữa f₁ GDP và f₂ bất bình đẳng trong tập Pareto là "
            f"<b>{corr_12:.3f}</b>. Nếu hệ số dương, tăng trưởng cao thường kéo theo bất bình đẳng vùng cao hơn, "
            "vì vốn có xu hướng chảy về ĐBSH/ĐNB nơi β_AI và β_D cao. Đây là đặc điểm chính sách quan trọng: "
            "đầu tư số thuần theo hiệu quả có thể làm rộng thêm khoảng cách số giữa vùng lõi và vùng yếu.",
        ),
        (
            "b)",
            "Trọng số (0,40; 0,25; 0,20; 0,15) có phù hợp ưu tiên Việt Nam?",
            "Bộ trọng số này đặt tăng trưởng lên hàng đầu nên phù hợp định hướng phát triển nhanh và nâng năng suất. "
            "Tuy nhiên, nếu nhấn mạnh COP26/net-zero 2050, nên tăng trọng số môi trường từ 0,20 lên 0,25–0,30. "
            "Nếu ưu tiên chủ quyền số theo Quyết định 127/QĐ-TTg, có thể tăng an ninh dữ liệu từ 0,15 lên 0,20–0,25.",
        ),
        (
            "c)",
            "NSGA-II khác gì LP đơn mục tiêu? Có thay thế quyết định chính trị không?",
            f"LP đơn mục tiêu trả về một nghiệm tối ưu duy nhất theo một hàm mục tiêu. NSGA-II trả về "
            f"<b>{n_p}</b> nghiệm Pareto, tức một bản đồ đánh đổi giữa tăng trưởng, bao trùm, môi trường và an ninh. "
            "NSGA-II không thay thế quyết định chính trị, vì lựa chọn nghiệm thỏa hiệp vẫn phụ thuộc vào trọng số, "
            "mức chấp nhận rủi ro, ưu tiên vùng miền và tính chính danh xã hội. Vai trò đúng của NSGA-II là làm rõ "
            "chi phí cơ hội của từng lựa chọn chính sách.",
        ),
    ]
    for code, q, ans in q_data:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()
