import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pricing Dataset", layout="wide")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def format_display_table(data: pd.DataFrame) -> pd.DataFrame:
    display = data.copy()

    # Format common numeric columns to match a "pricing table" look.
    currency_cols = [
        col
        for col in ["Price ($)", "Competitor Price ($)", "Revenue ($)"]
        if col in display.columns
    ]
    integer_cols = [col for col in ["Units Sold"] if col in display.columns]

    for col in currency_cols:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else ""
        )

    for col in integer_cols:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )

    return display


def build_page_list(current: int, total: int, window: int = 2) -> list[int | str]:
    if total <= 1:
        return [1]

    pages: list[int | str] = [1]
    start_p = max(2, current - window)
    end_p = min(total - 1, current + window)

    if start_p > 2:
        pages.append("...")

    pages.extend(range(start_p, end_p + 1))

    if end_p < total - 1:
        pages.append("...")

    if total > 1:
        pages.append(total)
    return pages


def goto(p: int, total_pages: int) -> None:
    st.session_state.page = max(1, min(p, total_pages))


st.markdown(
    """
    <style>
      .stApp {
        background: #f3f5f8;
      }

      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1500px;
        background: #ffffff;
        border: 1px solid #dfe4ea;
        border-radius: 8px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
      }

      .pricing-title {
        margin: 0;
        color: #163a6b;
        font-size: 2.05rem;
        font-weight: 700;
        line-height: 1.2;
      }

      .pricing-divider {
        border-top: 1px solid #e5e7eb;
        margin: 10px 0 14px 0;
      }

      .entries-label {
        color: #2a2f36;
        font-size: 0.98rem;
        margin: 2px 0 10px 0;
      }

      .pricing-table-wrap {
        border: 1px solid #d7dde5;
        border-radius: 4px;
        overflow-x: auto;
        background: #fff;
      }

      table.pricing-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.96rem;
      }

      table.pricing-table thead th {
        background: #f5f7fa;
        color: #1f2937;
        font-weight: 700;
        text-align: left;
        border-right: 1px solid #d7dde5;
        border-bottom: 1px solid #d7dde5;
        padding: 10px 12px;
        white-space: nowrap;
      }

      table.pricing-table tbody td {
        border-right: 1px solid #e1e6ee;
        border-bottom: 1px solid #e1e6ee;
        padding: 9px 12px;
        color: #111827;
        white-space: nowrap;
      }

      table.pricing-table thead th:last-child,
      table.pricing-table tbody td:last-child {
        border-right: none;
      }

      table.pricing-table tbody tr:nth-child(even) {
        background: #fcfdff;
      }

      /* Search input */
      [data-testid="stTextInput"] label {
        display: none;
      }
      [data-testid="stTextInput"] input {
        border: 1px solid #cfd6df;
        border-radius: 4px;
        padding: 0.55rem 0.75rem;
        min-height: 42px;
        box-shadow: none;
        background: #fff;
      }
      [data-testid="stTextInput"] input:focus {
        border-color: #8fb4e8;
        box-shadow: 0 0 0 1px #8fb4e8;
      }

      /* Buttons (pagination) */
      [data-testid="stButton"] > button {
        width: 100%;
        min-height: 38px;
        border-radius: 4px;
        border: 1px solid #d0d7e2;
        background: #ffffff;
        color: #4b5563;
        box-shadow: none;
        white-space: nowrap;
        padding: 0 8px;
      }

      [data-testid="stButton"] > button:hover {
        border-color: #9ab6db;
        color: #1f2937;
      }

      [data-testid="stButton"] > button[kind="primary"] {
        background: #2f78d3;
        border-color: #2f78d3;
        color: #ffffff;
      }

      .ellipsis-box {
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6b7280;
        border: 1px solid #d0d7e2;
        border-radius: 4px;
        background: #fff;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_PATH = "pricing_dataset.csv"
df = load_data(DATA_PATH)

if "page" not in st.session_state:
    st.session_state.page = 1
if "_last_query" not in st.session_state:
    st.session_state._last_query = ""


title_col, search_col = st.columns([4.2, 1.3])
with title_col:
    st.markdown('<h1 class="pricing-title">Pricing Dataset</h1>', unsafe_allow_html=True)
with search_col:
    q = st.text_input(
        "Search",
        value=st.session_state._last_query,
        placeholder="Search",
        label_visibility="collapsed",
        key="pricing_search",
    )

st.markdown('<div class="pricing-divider"></div>', unsafe_allow_html=True)

if q != st.session_state._last_query:
    st.session_state.page = 1
    st.session_state._last_query = q


# Search filter across all columns
filtered = df.copy()
if q.strip():
    mask = pd.Series(False, index=filtered.index)
    q_lower = q.lower()
    for col in filtered.columns:
        s = filtered[col].astype(str).str.lower()
        mask = mask | s.str.contains(q_lower, na=False, regex=False)
    filtered = filtered[mask]


# Pagination
PAGE_SIZE = 20
total_rows = len(filtered)
total_pages = max(1, math.ceil(total_rows / PAGE_SIZE))
st.session_state.page = max(1, min(st.session_state.page, total_pages))

start_idx = (st.session_state.page - 1) * PAGE_SIZE
end_idx = min(start_idx + PAGE_SIZE, total_rows)

showing_start = start_idx + 1 if total_rows else 0
showing_end = end_idx
entries_text = f"Showing {showing_start} to {showing_end} of {total_rows:,} entries"

st.markdown(f'<div class="entries-label">{entries_text}</div>', unsafe_allow_html=True)


# Table
page_df = filtered.iloc[start_idx:end_idx].copy()
display_df = format_display_table(page_df)

table_html = display_df.to_html(
    index=False,
    classes=["pricing-table"],
    border=0,
    escape=True,
)

st.markdown(f'<div class="pricing-table-wrap">{table_html}</div>', unsafe_allow_html=True)


# Footer: entries + pagination
st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
footer_left, footer_right = st.columns([2.3, 3.7])
with footer_left:
    st.markdown(f'<div class="entries-label" style="margin:0;">{entries_text}</div>', unsafe_allow_html=True)

with footer_right:
    pages = build_page_list(st.session_state.page, total_pages, window=2)
    items = ["prev", *pages, "next"]
    widths = []
    for item in items:
        if item in ("prev", "next"):
            widths.append(1.55)
        else:
            widths.append(0.72)

    cols = st.columns(widths, gap="small")
    for i, item in enumerate(items):
        with cols[i]:
            if item == "prev":
                st.button(
                    "Previous",
                    key="prev_btn",
                    on_click=goto,
                    args=(st.session_state.page - 1, total_pages),
                    disabled=(st.session_state.page <= 1),
                    use_container_width=True,
                )
            elif item == "next":
                st.button(
                    "Next",
                    key="next_btn",
                    on_click=goto,
                    args=(st.session_state.page + 1, total_pages),
                    disabled=(st.session_state.page >= total_pages),
                    use_container_width=True,
                )
            elif item == "...":
                st.markdown('<div class="ellipsis-box">...</div>', unsafe_allow_html=True)
            else:
                page_num = int(item)
                st.button(
                    str(page_num),
                    key=f"page_{page_num}",
                    on_click=goto,
                    args=(page_num, total_pages),
                    type="primary" if page_num == st.session_state.page else "secondary",
                    use_container_width=True,
                )
