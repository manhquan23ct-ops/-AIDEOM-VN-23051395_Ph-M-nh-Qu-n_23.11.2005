"""
data_loader.py
--------------
Load và cache toàn bộ dữ liệu CSV của project AIDEOM-VN.
Import từ bất kỳ page nào: from data_loader import load_macro, load_sectors, load_regions
"""

import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_macro() -> pd.DataFrame:
    """vietnam_macro_2020_2025.csv — 6 năm × 19 chỉ tiêu vĩ mô"""
    df = pd.read_csv(DATA_DIR / "vietnam_macro_2020_2025.csv")
    return df.sort_values("year").reset_index(drop=True)


@st.cache_data
def load_sectors() -> pd.DataFrame:
    """vietnam_sectors_2024.csv — 10 ngành × 13 chỉ tiêu"""
    return pd.read_csv(DATA_DIR / "vietnam_sectors_2024.csv")


@st.cache_data
def load_regions() -> pd.DataFrame:
    """vietnam_regions_2024.csv — 6 vùng × 14 chỉ tiêu"""
    return pd.read_csv(DATA_DIR / "vietnam_regions_2024.csv")


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Macro :", load_macro().shape)
    print("Sectors:", load_sectors().shape)
    print("Regions:", load_regions().shape)
    print(load_macro().head(2))
