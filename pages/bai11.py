"""pages/bai11.py — Bài 11: Q-learning cho chính sách kinh tế thích nghi

Bản làm lại:
- Có class VietnamEconomyEnv kế thừa gymnasium.Env đúng yêu cầu 11.3.1.
- Huấn luyện tabular Q-learning 10.000 episodes, alpha=0.1, gamma=0.95,
  epsilon giảm tuyến tính từ 1.00 xuống 0.05.
- Training dùng random initial states để học đủ không gian 81 trạng thái;
  evaluation dùng trạng thái Việt Nam 2026.
- Có so sánh π* với rule-based policies và phần mở rộng DQN.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import bai_header, end_padding, info_box, section_title

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except Exception:  # pragma: no cover - fallback cho máy chưa cài gymnasium
    HAS_GYM = False

    class _DummyEnv:
        metadata = {}
        def reset(self, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)

    class _Discrete:
        def __init__(self, n): self.n = int(n)
        def sample(self): return int(np.random.randint(self.n))

    class _MultiDiscrete:
        def __init__(self, nvec): self.nvec = np.array(nvec, dtype=int)
        def sample(self): return np.array([np.random.randint(n) for n in self.nvec])

    class gym:  # type: ignore
        Env = _DummyEnv

    class spaces:  # type: ignore
        Discrete = _Discrete
        MultiDiscrete = _MultiDiscrete


# ──────────────────────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"; C2 = "#2E8B57"; C3 = "#4CAF72"
CBLUE = "#1976D2"; CRED = "#E53935"; CORANGE = "#E65100"; CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

# 5 hành động theo đề: phân bổ K/D/AI/H
ALLOC = {
    0: np.array([0.70, 0.10, 0.10, 0.10], dtype=float),
    1: np.array([0.40, 0.25, 0.15, 0.20], dtype=float),
    2: np.array([0.25, 0.45, 0.15, 0.15], dtype=float),
    3: np.array([0.20, 0.20, 0.45, 0.15], dtype=float),
    4: np.array([0.30, 0.20, 0.10, 0.40], dtype=float),
}
ACT_NAMES = ["Truyền thống", "Cân bằng", "Số hóa nhanh", "AI dẫn dắt", "Bao trùm"]
ACT_DESC = [
    "70% K · 10% D · 10% AI · 10% H",
    "40% K · 25% D · 15% AI · 20% H",
    "25% K · 45% D · 15% AI · 15% H",
    "20% K · 20% D · 45% AI · 15% H",
    "30% K · 20% D · 10% AI · 40% H",
]
ACT_COLORS = ["#8AA898", CBLUE, C2, CRED, CPURPLE]
LEVELS = ["low", "medium", "high"]
U_LEVELS = ["low", "medium", "high"]

# Reward weights: ΔGDP, Δunemployment, cyber risk, emission
W_REWARD = np.array([0.40, 0.25, 0.20, 0.15], dtype=float)

# Cobb-Douglas parameters from Bài 1
ALPHA, BETA_L, GAMMA_D, DELTA_AI, THETA_H = 0.33, 0.42, 0.10, 0.08, 0.07


@dataclass
class InitialEconomy:
    K0: float = 27500.0
    D0: float = 20.3
    AI0: float = 86.0
    H0: float = 30.0
    L0: float = 53.9
    A0: float = 34.91


def _read_macro_initials() -> InitialEconomy:
    """Đọc CSV macro nếu có để hiệu chỉnh A0; fallback theo đề."""
    candidates = [
        "data/vietnam_macro_2020_2025.csv",
        "vietnam_macro_2020_2025.csv",
        "/mnt/data/vietnam_macro_2020_2025(1).csv",
        "/mnt/data/vietnam_macro_2020_2025.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Hỗ trợ một số tên cột thường gặp
                y_col = "GDP_trillion_VND" if "GDP_trillion_VND" in df.columns else None
                if y_col is None:
                    continue
                y = df[y_col].to_numpy(dtype=float)
                # Dùng chuỗi tham số trong đề Bài 1 nếu CSV không đủ cột
                K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
                L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], dtype=float)
                D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5], dtype=float)
                AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], dtype=float)
                H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], dtype=float)
                core = K**ALPHA * L**BETA_L * D**GAMMA_D * AI**DELTA_AI * H**THETA_H
                A = y[:6] / core
                A0 = float(A[-1] * 1.012)  # hiệu chỉnh 2026 nhẹ theo Bài 8
                return InitialEconomy(A0=A0)
            except Exception:
                pass
    return InitialEconomy()


INIT = _read_macro_initials()


class VietnamEconomyEnv(gym.Env):
    """MDP đơn giản hóa nền kinh tế Việt Nam cho tabular Q-learning.

    Observation: MultiDiscrete([3,3,3,3]) = GDP growth, Digital index, AI capacity, unemployment risk.
    Action: 5 chiến lược phân bổ ngân sách K/D/AI/H.
    Episode length: 10 năm.
    """

    metadata = {"render_modes": []}

    def __init__(self, random_start: bool = False, seed: int | None = None):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.T = 10
        self.random_start = bool(random_start)
        self.budget = 1000.0  # nghìn tỷ VND/năm, theo gợi ý đề
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def _production(self, K: float, D: float, AI: float, H: float, t: int = 0) -> float:
        L = INIT.L0 * (1.004 ** t)
        D_eff = np.clip(D, 5.0, 60.0)
        AI_eff = np.clip(AI, 20.0, 320.0)
        H_eff = np.clip(H, 10.0, 95.0)
        A = INIT.A0 * (1.006 ** t)
        return float(A * K**ALPHA * L**BETA_L * D_eff**GAMMA_D * AI_eff**DELTA_AI * H_eff**THETA_H)

    @staticmethod
    def _state_from_values(growth_pct: float, D: float, AI: float, U: float) -> np.ndarray:
        g_level = 0 if growth_pct < 4.0 else (1 if growth_pct < 7.0 else 2)
        d_level = 0 if D < 25.0 else (1 if D < 40.0 else 2)
        ai_level = 0 if AI < 120.0 else (1 if AI < 190.0 else 2)
        u_level = 0 if U < 3.0 else (1 if U < 5.0 else 2)
        return np.array([g_level, d_level, ai_level, u_level], dtype=np.int64)

    @staticmethod
    def _values_from_state(state: Tuple[int, int, int, int]) -> Tuple[float, float, float, float, float]:
        """Tạo điểm đại diện liên tục cho một trạng thái rời rạc."""
        g, d, ai, u = state
        D = [20.0, 32.0, 45.0][d]
        AI = [90.0, 155.0, 230.0][ai]
        H = [32.0, 50.0, 72.0][2 - u]  # U cao ↔ H thấp
        U = [2.5, 4.0, 6.5][u]
        K = [25000.0, 28500.0, 32500.0][g]
        return K, D, AI, H, U

    def reset(self, seed=None, options=None):  # noqa: D401
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        state_opt = options.get("state")
        random_start = options.get("random_start", self.random_start)

        self.t = 0
        if state_opt is not None:
            s = tuple(int(v) for v in state_opt)
            self.K, self.D, self.AI, self.H, self.U = self._values_from_state(s)
            self.state = np.array(s, dtype=np.int64)
        elif random_start:
            self.state = self.rng.integers(0, 3, size=4, dtype=np.int64)
            self.K, self.D, self.AI, self.H, self.U = self._values_from_state(tuple(self.state))
        else:
            self.K, self.D, self.AI, self.H = INIT.K0, INIT.D0, INIT.AI0, INIT.H0
            self.U = 4.0
            Y0 = self._production(self.K, self.D, self.AI, self.H, 0)
            Y1 = self._production(self.K * 1.045, self.D, self.AI, self.H, 1)
            self.state = self._state_from_values((Y1 / Y0 - 1) * 100, self.D, self.AI, self.U)

        self.Y = self._production(self.K, self.D, self.AI, self.H, self.t)
        return self.state.copy(), {}

    def step(self, action: int):
        action = int(action)
        a = ALLOC[action]
        prev_Y = self.Y
        prev_U = self.U

        # Chuyển trạng thái: có khấu hao vốn, D/AI/H tăng theo hệ số chuyển đổi đơn vị
        self.K = 0.95 * self.K + a[0] * self.budget
        self.D = float(np.clip(0.985 * self.D + a[1] * self.budget / 100.0, 5.0, 60.0))
        self.AI = float(np.clip(0.97 * self.AI + a[2] * self.budget / 20.0, 20.0, 320.0))
        self.H = float(np.clip(0.99 * self.H + a[3] * self.budget / 200.0, 10.0, 95.0))

        # Thất nghiệp giảm khi H tăng và GDP tăng, tăng khi tự động hóa AI quá mạnh
        self.Y = self._production(self.K, self.D, self.AI, self.H, self.t + 1)
        g_pct = (self.Y / max(prev_Y, 1e-9) - 1.0) * 100.0
        auto_pressure = 1.2 * a[2]
        h_absorb = 1.4 * a[3]
        growth_absorb = 0.06 * g_pct
        self.U = float(np.clip(prev_U + auto_pressure - h_absorb - growth_absorb, 1.2, 9.0))

        # Rủi ro an ninh & phát thải: AI/D làm tăng cyber risk; K và AI làm tăng emission gián tiếp
        cyber_risk = 2.0 * a[2] + 0.8 * a[1] - 0.4 * a[3]
        cyber_risk = max(cyber_risk, 0.0)
        emission = 1.5 * a[0] + 0.8 * a[2] + 0.3 * a[1]
        delta_u = self.U - prev_U

        reward = (
            W_REWARD[0] * g_pct
            - W_REWARD[1] * delta_u * 10.0
            - W_REWARD[2] * cyber_risk
            - W_REWARD[3] * emission
        )

        self.t += 1
        self.state = self._state_from_values(g_pct, self.D, self.AI, self.U)
        terminated = self.t >= self.T
        truncated = False
        info = {
            "Y": self.Y,
            "growth_pct": g_pct,
            "U": self.U,
            "cyber_risk": cyber_risk,
            "emission": emission,
            "K": self.K,
            "D": self.D,
            "AI": self.AI,
            "H": self.H,
        }
        return self.state.copy(), float(reward), terminated, truncated, info


@st.cache_data(show_spinner=False)
def train_q_learning(
    n_episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float], List[float], np.ndarray]:
    """Tabular Q-learning. Random start giúp học chính sách cho cả 81 trạng thái."""
    rng = np.random.default_rng(seed)
    env = VietnamEconomyEnv(random_start=True, seed=seed)
    Q = np.zeros((3, 3, 3, 3, 5), dtype=float)
    visits = np.zeros((3, 3, 3, 3), dtype=int)
    ep_rewards: List[float] = []
    eps_history: List[float] = []

    for ep in range(n_episodes):
        # epsilon giảm tuyến tính từ 1.00 xuống 0.05 qua đúng 10.000 episodes
        eps = max(0.05, 1.0 - 0.95 * ep / max(n_episodes - 1, 1))
        eps_history.append(float(eps))
        state, _ = env.reset(seed=int(rng.integers(0, 2**32 - 1)), options={"random_start": True})
        s = tuple(int(v) for v in state)
        total_r = 0.0
        done = False
        while not done:
            visits[s] += 1
            if rng.random() < eps:
                action = int(rng.integers(0, 5))
            else:
                action = int(np.argmax(Q[s]))
            state2, reward, terminated, truncated, _ = env.step(action)
            s2 = tuple(int(v) for v in state2)
            td_target = reward + gamma * np.max(Q[s2]) * (0.0 if terminated else 1.0)
            Q[s + (action,)] += alpha * (td_target - Q[s + (action,)])
            s = s2
            total_r += reward
            done = terminated or truncated
        ep_rewards.append(float(total_r))
    return Q, ep_rewards, eps_history, visits


def evaluate_policy(policy_fn: Callable[[Tuple[int, int, int, int], np.random.Generator], int], n: int = 200, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    rewards: List[float] = []
    for k in range(n):
        env = VietnamEconomyEnv(random_start=False, seed=int(rng.integers(0, 2**32 - 1)))
        state, _ = env.reset(options={"random_start": False})
        s = tuple(int(v) for v in state)
        total_r = 0.0
        done = False
        while not done:
            action = policy_fn(s, rng)
            state, reward, terminated, truncated, _ = env.step(action)
            s = tuple(int(v) for v in state)
            total_r += reward
            done = terminated or truncated
        rewards.append(float(total_r))
    return float(np.mean(rewards)), float(np.std(rewards))


def rollout(policy_fn: Callable[[Tuple[int, int, int, int], np.random.Generator], int], seed: int = 999) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    env = VietnamEconomyEnv(random_start=False, seed=seed)
    state, _ = env.reset(options={"random_start": False})
    rows = []
    done = False
    year = 2026
    while not done:
        s = tuple(int(v) for v in state)
        a = policy_fn(s, rng)
        state2, reward, terminated, truncated, info = env.step(a)
        rows.append({
            "Năm": year,
            "State": f"({LEVELS[s[0]]}, {LEVELS[s[1]]}, {LEVELS[s[2]]}, {U_LEVELS[s[3]]})",
            "Action": f"a{a} — {ACT_NAMES[a]}",
            "Reward": reward,
            "Y": info["Y"],
            "Growth %": info["growth_pct"],
            "U risk": info["U"],
            "D": info["D"],
            "AI": info["AI"],
            "H": info["H"],
        })
        state = state2
        year += 1
        done = terminated or truncated
    return pd.DataFrame(rows)


def render():
    bai_header(
        so="11",
        ten="Học tăng cường Q-learning cho chính sách kinh tế thích nghi",
        mo_ta="MDP 81 trạng thái × 5 hành động, Q-learning tabular 10.000 episodes, so sánh π* với rule-based",
        cap_do="KHÓ",
        tools=["gymnasium", "Q-learning", "plotly"],
        thoi_luong="3-4 tuần",
    )

    info_box(
        "MDP: <b>S = {low, medium, high}⁴ = 81 trạng thái</b> gồm GDP growth, Digital Index, AI capacity, Unemployment risk; "
        "<b>A = 5 chiến lược ngân sách</b>.<br>"
        "Reward: <b>R = 0,40·ΔGDP − 0,25·ΔU − 0,20·CyberRisk − 0,15·Emission</b>. "
        "Q-learning: α=0,10 · γ=0,95 · ε giảm 1,00 → 0,05 qua 10.000 episodes.",
        bg="#E8F5E9", border=C1, icon="📐",
    )

    if not HAS_GYM:
        info_box(
            "Máy chưa cài <code>gymnasium</code>. File vẫn chạy bằng fallback nội bộ, nhưng để đúng yêu cầu đề nên cài: "
            "<code>pip install gymnasium</code>.",
            bg="#FFF8E1", border=CORANGE, icon="⚠️",
        )

    # ── 11.3.1 Environment ─────────────────────────────────────────────────
    section_title("Câu 11.3.1 — Môi trường Gymnasium MDP", "🏗️")
    col_env, col_act = st.columns([2, 3], gap="large")
    with col_env:
        df_state = pd.DataFrame({
            "Thành phần trạng thái": ["GDP growth", "Digital index", "AI capacity", "Unemployment risk"],
            "Mức 0": ["low", "low", "low", "low"],
            "Mức 1": ["medium", "medium", "medium", "medium"],
            "Mức 2": ["high", "high", "high", "high"],
        })
        st.dataframe(df_state, use_container_width=True, hide_index=True)
        info_box("Tổng số trạng thái: <b>3⁴ = 81</b>. Một episode mô phỏng <b>10 năm</b>.", bg="#F1F8F2", border=C2, icon="✅")
    with col_act:
        df_act = pd.DataFrame({
            "Action": [f"a{i}" for i in range(5)],
            "Tên": ACT_NAMES,
            "K/D/AI/H": ACT_DESC,
            "Ý nghĩa": [
                "Ưu tiên vốn vật chất truyền thống",
                "Cân bằng tăng trưởng và ổn định",
                "Tăng tốc hạ tầng số và CĐS",
                "Đặt AI làm động lực chính",
                "Ưu tiên nhân lực, giảm rủi ro thất nghiệp",
            ],
        })
        st.dataframe(df_act, use_container_width=True, hide_index=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── 11.3.2 Training ─────────────────────────────────────────────────────
    section_title("Câu 11.3.2 — Huấn luyện Q-learning", "🧠")
    with st.spinner("Đang huấn luyện Q-learning 10.000 episodes..."):
        Q, ep_rewards, eps_history, visits = train_q_learning(10000, 0.1, 0.95, 42)

    avg_first = float(np.mean(ep_rewards[:200]))
    avg_last = float(np.mean(ep_rewards[-200:]))
    visited_states = int(np.sum(visits > 0))
    greedy_policy = np.argmax(Q, axis=-1)

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    for col, lbl, val, sub, color in [
        (k1, "Episodes", "10.000", "T = 10 năm/episode", C1),
        (k2, "Reward đầu", f"{avg_first:.2f}", "200 episodes đầu", "#8AA898"),
        (k3, "Reward cuối", f"{avg_last:.2f}", f"Δ={avg_last-avg_first:+.2f}", CBLUE),
        (k4, "States visited", f"{visited_states}/81", "random-start training", CPURPLE),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:0.78rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
                <div style="font-size:1.7rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                <div style="font-size:0.78rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Learning curve
    window = 250
    smooth = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(
        x=np.arange(window - 1, len(ep_rewards)), y=smooth,
        name=f"Reward MA-{window}", line=dict(color=C1, width=2.5),
    ))
    fig_lc.add_trace(go.Scatter(
        x=np.arange(len(eps_history)), y=np.array(eps_history) * max(np.max(smooth), 1) / 1.1,
        name="ε scaled", yaxis="y2", line=dict(color=CORANGE, width=2, dash="dot"),
    ))
    fig_lc.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CF,
        legend=dict(orientation="h", y=-0.22),
        yaxis=dict(showgrid=True, gridcolor="#F0F4F0", title="Reward/episode"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="ε scaled"),
        xaxis=dict(showgrid=False, title="Episode"),
    )
    st.plotly_chart(fig_lc, use_container_width=True, config={"displayModeBar": False})
    info_box(
        "Khác file cũ: training dùng <b>random initial states</b> để học đủ chính sách cho các trạng thái giả định, "
        "không chỉ học quỹ đạo từ Việt Nam 2026. Vì vậy bảng π*(s) ở câu 11.3.3 không còn bị 'chưa thăm' ở các trạng thái cực đoan.",
        bg="#E3F2FD", border=CBLUE, icon="📋",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── 11.3.3 Policy extraction ────────────────────────────────────────────
    section_title("Câu 11.3.3 — Trích xuất chính sách π*(s)", "🗺️")
    key_states = [
        ("🇻🇳 Việt Nam 2026", (1, 1, 0, 1)),
        ("📉 GDP thấp · D thấp · AI thấp · U cao", (0, 0, 0, 2)),
        ("🚀 GDP cao · D cao · AI cao · U thấp", (2, 2, 2, 0)),
        ("⚖️ Mọi chỉ số medium", (1, 1, 1, 1)),
        ("🌱 GDP thấp · D thấp · AI medium · U medium", (0, 0, 1, 1)),
    ]
    rows = []
    for label, s in key_states:
        a = int(greedy_policy[s])
        rows.append({
            "Trạng thái": label,
            "(GDP,D,AI,U)": f"({LEVELS[s[0]]}, {LEVELS[s[1]]}, {LEVELS[s[2]]}, {U_LEVELS[s[3]]})",
            "π*(s)": f"a{a} — {ACT_NAMES[a]}",
            "Q*": f"{Q[s][a]:.3f}",
            "Visits": int(visits[s]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    s_vn = (1, 1, 0, 1)
    fig_q = go.Figure(go.Bar(
        x=[f"a{i} — {ACT_NAMES[i]}" for i in range(5)],
        y=[Q[s_vn][i] for i in range(5)],
        marker_color=ACT_COLORS, marker_line_width=0,
        text=[f"{Q[s_vn][i]:.2f}" for i in range(5)], textposition="outside",
    ))
    best_vn = int(greedy_policy[s_vn])
    fig_q.add_annotation(x=best_vn, y=Q[s_vn][best_vn], text="⭐ π*", showarrow=True, arrowhead=2, ax=0, ay=-30)
    fig_q.update_layout(
        height=280, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CF,
        showlegend=False, yaxis=dict(showgrid=True, gridcolor="#F0F4F0"), xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_q, use_container_width=True, config={"displayModeBar": False})
    info_box(
        f"Tại trạng thái Việt Nam 2026, π* chọn <b>a{best_vn} — {ACT_NAMES[best_vn]}</b>. "
        "Diễn giải cần gắn với trạng thái: AI capacity còn thấp nhưng D ở mức trung bình, nên agent thường ưu tiên hành động mở khóa năng lực tương lai, "
        "đồng thời vẫn cân bằng rủi ro thất nghiệp và an ninh dữ liệu.",
        bg="#FFF8E1", border=CORANGE, icon="💡",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── 11.3.4 Policy comparison ────────────────────────────────────────────
    section_title("Câu 11.3.4 — So sánh π* với rule-based policies", "📊")
    policies: Dict[str, Callable[[Tuple[int, int, int, int], np.random.Generator], int]] = {
        "π* Q-learning": lambda s, rng: int(greedy_policy[s]),
        "Luôn a1 — Cân bằng": lambda s, rng: 1,
        "Luôn a3 — AI dẫn dắt": lambda s, rng: 3,
        "Random": lambda s, rng: int(rng.integers(0, 5)),
    }
    results = {name: evaluate_policy(fn, n=200, seed=123) for name, fn in policies.items()}
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    order = np.argsort(-np.array(means))
    fig_cmp = go.Figure(go.Bar(
        x=[names[i] for i in order], y=[means[i] for i in order],
        error_y=dict(type="data", array=[stds[i] for i in order]),
        marker_color=[C1, CBLUE, CRED, "#8AA898"], marker_line_width=0,
        text=[f"{means[i]:.2f}" for i in order], textposition="outside",
    ))
    fig_cmp.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CF,
        showlegend=False, yaxis=dict(showgrid=True, gridcolor="#F0F4F0", title="Reward tích lũy"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

    best_mean = results["π* Q-learning"][0]
    df_res = pd.DataFrame({
        "Chính sách": names,
        "Reward TB": [f"{results[n][0]:.3f}" for n in names],
        "Std": [f"±{results[n][1]:.3f}" for n in names],
        "So với π*": [f"{(results[n][0] - best_mean) / max(abs(best_mean), 1e-9) * 100:+.1f}%" for n in names],
    })
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    df_roll = rollout(lambda s, rng: int(greedy_policy[s]), seed=2026)
    with st.expander("Xem rollout 10 năm của π* từ trạng thái Việt Nam 2026", expanded=False):
        st.dataframe(df_roll.round({"Reward": 3, "Y": 1, "Growth %": 2, "U risk": 2, "D": 2, "AI": 2, "H": 2}), use_container_width=True, hide_index=True)

    info_box(
        "π* không nhất thiết luôn chọn một action cố định. Lợi thế của RL là <b>state-contingent policy</b>: "
        "khi U cao có thể nghiêng về Bao trùm; khi nền tảng đã tốt có thể chuyển sang AI hoặc Số hóa nhanh. "
        "Đây là điểm khác với LP tĩnh/rule-based.",
        bg="#E8F5E9", border=C3, icon="🏆",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── 11.3.5 DQN ──────────────────────────────────────────────────────────
    section_title("Câu 11.3.5 ★ — Mở rộng Deep Q-Network (DQN)", "🧬")
    info_box(
        "Với bài này chỉ có <b>81 trạng thái</b>, tabular Q-learning là phù hợp và dễ giải thích hơn DQN. "
        "DQN chỉ nên dùng khi trạng thái liên tục hoặc số trạng thái tăng rất lớn.<br><br>"
        "Code mẫu: <code>from stable_baselines3 import DQN</code>; "
        "<code>model = DQN('MlpPolicy', env, policy_kwargs={'net_arch':[64,64]}, learning_rate=1e-3)</code>; "
        "<code>model.learn(total_timesteps=50000)</code>.",
        bg="#F3E5F5", border=CPURPLE, icon="🧬",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── Policy discussion ───────────────────────────────────────────────────
    section_title("Câu hỏi thảo luận chính sách", "💬")
    bad_state = (0, 0, 0, 2)
    good_state = (2, 2, 2, 0)
    a_bad = int(greedy_policy[bad_state])
    a_good = int(greedy_policy[good_state])
    q_data = [
        (
            "a)",
            "GDP thấp, D thấp, U cao → π*(s) chọn gì? Có khớp 'quick win' không?",
            f"Trong mô hình đã huấn luyện, π* chọn <b>a{a_bad} — {ACT_NAMES[a_bad]}</b>. "
            "Nếu trạng thái xấu kèm thất nghiệp cao, chính sách hợp lý thường là ưu tiên H/bao trùm hoặc số hóa nhanh có kiểm soát: "
            "vừa tạo năng lực hấp thụ, vừa tránh đẩy AI quá mạnh khi thị trường lao động chưa sẵn sàng. Đây là logic 'quick win' xã hội, không chỉ quick win GDP.",
        ),
        (
            "b)",
            "GDP cao, AI cao, U thấp → chính sách chọn gì? Có phù hợp 'consolidation' không?",
            f"π* chọn <b>a{a_good} — {ACT_NAMES[a_good]}</b>. Với nền kinh tế đã ở trạng thái tốt, ý nghĩa chính sách là củng cố thành quả: "
            "không nhất thiết tiếp tục tăng AI cực mạnh, mà cân bằng giữa tăng trưởng, an ninh dữ liệu, phát thải và duy trì kỹ năng lao động.",
        ),
        (
            "c)",
            "Tích hợp π* vào quy trình hoạch định chính sách thế nào để không thay thế quyết định chính trị?",
            "Dùng nguyên tắc <b>AI-as-Advisor, Human-as-Decider</b>: π* chỉ là gợi ý trong dashboard; mỗi khuyến nghị phải kèm Q-value, trạng thái đầu vào, giả định mô hình, "
            "và so sánh với chính sách thay thế. Quyết định cuối phải do cơ quan có thẩm quyền phê duyệt, có audit log, phản biện chuyên gia, đánh giá tác động xã hội và cơ chế khiếu nại. "
            "Như vậy RL hỗ trợ minh bạch hóa đánh đổi, không tự động hóa quyền lực chính trị.",
        ),
    ]
    for code, q, ans in q_data:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()
