import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import html
from pathlib import Path

from train_pricing_model import prepare_training_data, train_and_evaluate

st.set_page_config(layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("pricing_dataset.csv")


@st.cache_resource
def load_trained_bundle(data_path: str, file_mtime: float):
    # file_mtime is part of the cache key to refresh if the CSV changes.
    df_model = pd.read_csv(data_path)
    pipeline, model_metrics, model_metadata = train_and_evaluate(df_model, target="Units Sold")
    X_model, y_model = prepare_training_data(df_model, "Units Sold")
    return pipeline, model_metrics, model_metadata, X_model, y_model


def build_baseline_feature_row(X_model: pd.DataFrame) -> pd.Series:
    base = X_model.iloc[0].copy()
    for col in X_model.columns:
        col_series = X_model[col]
        if pd.api.types.is_numeric_dtype(col_series):
            med = pd.to_numeric(col_series, errors="coerce").median()
            if pd.notna(med):
                base[col] = float(med)
        else:
            modes = col_series.mode(dropna=True)
            if not modes.empty:
                base[col] = modes.iloc[0]
    return base


def make_price_scenarios(X_model: pd.DataFrame, prices: np.ndarray) -> pd.DataFrame:
    base = build_baseline_feature_row(X_model)
    scenario_df = pd.DataFrame([base.to_dict() for _ in range(len(prices))])

    if "Price ($)" in scenario_df.columns:
        scenario_df["Price ($)"] = prices

    if "Competitor Price ($)" in scenario_df.columns and "Price ($)" in X_model.columns:
        gap = (
            pd.to_numeric(X_model["Competitor Price ($)"], errors="coerce")
            - pd.to_numeric(X_model["Price ($)"], errors="coerce")
        ).median()
        gap = float(gap) if pd.notna(gap) else 0.0
        scenario_df["Competitor Price ($)"] = scenario_df["Price ($)"] + gap

    return scenario_df


def local_elasticity(prices: np.ndarray, demand: np.ndarray, idx: int) -> float:
    if len(prices) < 3:
        return 0.0
    i = max(1, min(int(idx), len(prices) - 2))
    dqdP = (float(demand[i + 1]) - float(demand[i - 1])) / (float(prices[i + 1]) - float(prices[i - 1]))
    q = max(float(demand[i]), 1e-6)
    p = float(prices[i])
    return float(dqdP * (p / q))

df = load_data()

st.title("📊 Dynamic Pricing Dashboard")

st.markdown(
    """
    <style>
      .stApp {
        background: #f3f5f9;
        color: #243b53;
      }

      /* Force readable text in the main page even if Streamlit theme is dark */
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #163d6d !important;
      }

      .stApp p,
      .stApp li,
      .stApp label,
      .stApp small {
        color: #243b53;
      }

      div[data-testid="stMarkdownContainer"] h1,
      div[data-testid="stMarkdownContainer"] h2,
      div[data-testid="stMarkdownContainer"] h3,
      div[data-testid="stMarkdownContainer"] h4,
      div[data-testid="stMarkdownContainer"] p,
      div[data-testid="stMarkdownContainer"] li {
        color: #243b53;
      }

      .metrics-showcase {
        margin-top: 1.25rem;
        background: #f1f4f8;
        border: 1px solid #e4eaf2;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
        padding: 1.2rem 1.2rem 1.35rem 1.2rem;
      }

      .metrics-showcase-title {
        margin: 0;
        color: #163d6d !important;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.3px;
      }

      .metrics-showcase-divider {
        border-top: 1px solid #e5ebf2;
        margin: 0.95rem 0 1.1rem 0;
      }

      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1.1rem;
      }

      .metric-box {
        background: #fff;
        border: 1px solid #e2e8f1;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        overflow: hidden;
      }

      .metric-box-head {
        color: #ffffff !important;
        font-size: 1.15rem;
        font-weight: 600;
        padding: 0.8rem 1rem;
        letter-spacing: 0.2px;
      }

      .metric-box-head.blue {
        background: linear-gradient(135deg, #4f8ef7, #2d6bd0);
      }
      .metric-box-head.orange {
        background: linear-gradient(135deg, #f6ad3b, #f08b23);
      }
      .metric-box-head.green {
        background: linear-gradient(135deg, #4dbb99, #38a37f);
      }
      .metric-box-head.sky {
        background: linear-gradient(135deg, #8cbcf6, #6ba4ea);
      }

      .metric-box-body {
        padding: 1.05rem 1rem 0.95rem 1rem;
      }

      .metric-main-value {
        margin: 0;
        color: #153f73;
        font-size: 2.85rem;
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: 0.2px;
      }

      .metric-main-value.green {
        color: #1f7a63;
      }

      .metric-delta {
        margin-top: 0.55rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: #2f8b57;
        display: flex;
        align-items: center;
        gap: 0.35rem;
      }

      .metric-delta.negative {
        color: #c04545;
      }

      .metric-arrow {
        font-size: 0.95rem;
        line-height: 1;
      }

      .perf-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.75rem;
        color: #173f73;
        font-size: 1.05rem;
        font-weight: 500;
      }

      .perf-row strong {
        font-weight: 700;
      }

      .perf-row .big {
        font-size: 1.55rem;
        font-weight: 700;
      }

      .perf-divider {
        border-top: 1px solid #e6ebf2;
        margin: 0.9rem 0 0.75rem 0;
      }

      .perf-status {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #de8b2a;
        font-size: 1.05rem;
        font-weight: 500;
      }

      .perf-check {
        width: 1.9rem;
        height: 1.9rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        background: #ffffff;
        border: 1px solid #eceff5;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        color: #de8b2a;
        font-weight: 700;
      }

      .section-title {
        margin: 1rem 0 0.5rem 0;
        color: #163d6d;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.2px;
      }

      .section-subtitle {
        margin: 0 0 0.55rem 0;
        color: #5a6f86;
        font-size: 0.9rem;
      }

      /* Style Streamlit metric cards (top row) */
      [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f1;
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
      }
      [data-testid="stMetricLabel"] p {
        color: #52667d !important;
        font-weight: 600;
      }
      [data-testid="stMetricValue"] {
        color: #163d6d !important;
      }
      [data-testid="stMetricValue"] * {
        color: #163d6d !important;
      }
      [data-testid="stMetricDelta"] * {
        color: inherit !important;
      }
      [data-testid="stMetricDelta"] {
        font-weight: 600;
      }

      /* Style plot containers */
      div[data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e2e8f1;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        padding: 0.3rem 0.35rem 0.1rem 0.35rem;
      }

      /* Fallback in case Plotly inherits dark-theme text from Streamlit */
      div[data-testid="stPlotlyChart"] .js-plotly-plot text {
        fill: #243b53 !important;
      }
      div[data-testid="stPlotlyChart"] .gtitle {
        fill: #163d6d !important;
      }

      .sim-card {
        margin-top: 0.35rem;
        background: #ffffff;
        border: 1px solid #e2e8f1;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        overflow: hidden;
      }

      .sim-card-head {
        background: linear-gradient(135deg, #4f8ef7, #2d6bd0);
        color: white !important;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.7rem 0.9rem;
      }

      table.sim-table-custom {
        width: 100%;
        border-collapse: collapse;
        background: #fff;
      }

      table.sim-table-custom thead th {
        background: #f6f9fd;
        color: #39536d;
        font-size: 0.88rem;
        text-align: left;
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid #e5ebf2;
        font-weight: 700;
      }

      table.sim-table-custom tbody td {
        padding: 0.55rem 0.8rem;
        border-bottom: 1px solid #edf2f8;
        color: #243b53;
        font-size: 0.9rem;
      }

      table.sim-table-custom tbody tr:last-child td {
        border-bottom: none;
      }

      table.sim-table-custom tbody tr.highlight td {
        background: #fff7e3;
        font-weight: 600;
      }

      @media (max-width: 900px) {
        .metrics-grid {
          grid-template-columns: 1fr;
        }
        .metric-main-value {
          font-size: 2.2rem;
        }
        .perf-row {
          flex-direction: column;
          align-items: flex-start;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_plotly(fig, height: int = 320):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=18, t=52, b=24),
        font=dict(color="#243b53"),
        title=dict(font=dict(color="#163d6d", size=16), x=0.02, xanchor="left"),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", borderwidth=0),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#dbe5f0",
        tickfont=dict(color="#243b53"),
        title_font=dict(color="#163d6d"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e9eef5",
        zeroline=False,
        linecolor="#dbe5f0",
        tickfont=dict(color="#243b53"),
        title_font=dict(color="#163d6d"),
    )
    return fig


def render_scenarios_table(table_df: pd.DataFrame) -> None:
    rows = []
    for _, row in table_df.iterrows():
        cls = "highlight" if str(row["Scenario"]).lower().startswith("optim") else ""
        rows.append(
            "<tr class='{cls}'><td>{s}</td><td>${p:,.2f}</td><td>{d:,}</td><td>${r:,.0f}</td></tr>".format(
                cls=cls,
                s=html.escape(str(row["Scenario"])),
                p=float(row["Price"]),
                d=int(row["Demand"]),
                r=float(row["Revenue"]),
            )
        )

    table_html = textwrap.dedent(
        f"""
        <div class="sim-card">
          <div class="sim-card-head">Price Simulation &amp; Revenue Distribution</div>
          <table class="sim-table-custom">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Price</th>
                <th>Demand</th>
                <th>Revenue</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </div>
        """
    ).strip()
    st.markdown(table_html, unsafe_allow_html=True)

# =========================
# MODEL WITH PREPROCESSING PIPELINE
# =========================
data_path = "pricing_dataset.csv"
csv_mtime = Path(data_path).stat().st_mtime if Path(data_path).exists() else 0.0
trained_pipeline, trained_metrics, trained_metadata, X_model, y_model = load_trained_bundle(
    data_path, csv_mtime
)

X = X_model
y = y_model

price_num = pd.to_numeric(df["Price ($)"], errors="coerce")
price_min = float(price_num.min())
price_max = float(price_num.max())
price_range = np.linspace(price_min, price_max, 60)

price_curve_features = make_price_scenarios(X_model, price_range)
predicted_units = np.clip(trained_pipeline.predict(price_curve_features), 0, None)
revenue_curve = price_range * predicted_units
opt_idx = int(np.argmax(revenue_curve))

optimal_price = float(price_range[opt_idx])
predicted_demand = int(round(float(predicted_units[opt_idx])))
revenue_projection = round(float(revenue_curve[opt_idx]), 2)
elasticity = local_elasticity(price_range, predicted_units, opt_idx)


def predict_demand_for_price(price_value: float) -> int:
    features = make_price_scenarios(X_model, np.array([float(price_value)]))
    pred = float(trained_pipeline.predict(features)[0])
    return int(max(round(pred), 0))

# =========================
# KPI ROW
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Demand", f"{predicted_demand:,} Units")
col2.metric("Price Recommendation", f"${optimal_price:.2f}")
col3.metric("Revenue Projection", f"${revenue_projection:,.0f}")
col4.metric("Demand Elasticity", f"{elasticity:.2f}")

st.markdown("---")

# =========================
# PRICE OPTIMIZATION CURVE
# =========================
fig_opt = go.Figure()
fig_opt.add_trace(go.Scatter(x=price_range, y=predicted_units,
                             mode='lines', name='Demand Curve',
                             line=dict(color="#2d6bd0", width=3)))

fig_opt.add_trace(go.Scatter(
    x=[optimal_price],
    y=[predicted_demand],
    mode='markers',
    marker=dict(size=12, color="#f08b23", line=dict(color="#ffd39a", width=2)),
    name="Optimal Price"
))

fig_opt.update_layout(title="Price Optimization",
                      xaxis_title="Price ($)",
                      yaxis_title="Predicted Demand")
style_plotly(fig_opt, height=360)

# =========================
# DEMAND FORECAST
# =========================
daily = df.groupby("Date")["Units Sold"].sum().reset_index()

fig_demand = px.line(daily.tail(30),
                     x="Date",
                     y="Units Sold",
                     title="Demand Forecast & Actuals (Last 30 Days)")
fig_demand.update_traces(line=dict(color="#2d6bd0", width=3))
style_plotly(fig_demand, height=360)

# =========================
# SEASONAL IMPACT
# =========================
seasonal = df.groupby("Season")["Units Sold"].mean().reset_index()
fig_season = px.bar(seasonal,
                    x="Season",
                    y="Units Sold",
                    title="Seasonal Impact")
fig_season.update_traces(marker_color="#4f8ef7", marker_line_color="#2d6bd0", marker_line_width=1)
style_plotly(fig_season, height=320)

# =========================
# COMPETITOR PRICING
# =========================
fig_comp = px.scatter(df.sample(1000),
                      x="Price ($)",
                      y="Competitor Price ($)",
                      title="Competitor Pricing",
                      opacity=0.5)
fig_comp.update_traces(marker=dict(color="#38a37f", line=dict(width=0)))
style_plotly(fig_comp, height=320)

# =========================
# FEATURE IMPORTANCE (Correlation)
# =========================
corr = df.corr(numeric_only=True)["Units Sold"].sort_values(ascending=False)
corr = corr.drop("Units Sold")

fig_feat = px.bar(
    x=corr.values,
    y=corr.index,
    orientation='h',
    title="Feature Importance (Correlation)"
)
fig_feat.update_traces(marker_color="#6ba4ea", marker_line_color="#2d6bd0", marker_line_width=1)
style_plotly(fig_feat, height=320)

# =========================
# REVENUE DISTRIBUTION
# =========================
fig_rev = px.histogram(df,
                       x="Revenue ($)",
                       nbins=40,
                       title="Revenue Distribution")
fig_rev.update_traces(marker_color="#4dbb99", marker_line_color="#ffffff", marker_line_width=0.4)
style_plotly(fig_rev, height=320)

# =========================
# PRICE SIMULATION TABLE
# =========================
scenarios = pd.DataFrame({
    "Scenario": ["Baseline", "Optimized", "High Price", "Low Price"],
    "Price": [29.99, optimal_price, 49.99, 19.99]
})

scenarios["Demand"] = scenarios["Price"].apply(predict_demand_for_price)
scenarios["Revenue"] = scenarios["Price"] * scenarios["Demand"]

# =========================
# LAYOUT
# =========================
st.markdown('<div class="section-title">Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-subtitle">Version restylée + modele integre ({trained_metadata.get("model_type", "Pipeline")}) avec preprocessing (imputation + one-hot). R²={trained_metrics.get("r2", 0):.2f}, RMSE={trained_metrics.get("rmse", 0):.1f}.</div>',
    unsafe_allow_html=True
)

row1_col1, row1_col2 = st.columns(2)
row1_col1.plotly_chart(fig_opt, use_container_width=True)
row1_col2.plotly_chart(fig_demand, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)
row2_col1.plotly_chart(fig_season, use_container_width=True)
row2_col2.plotly_chart(fig_comp, use_container_width=True)

row3_col1, row3_col2 = st.columns(2)
row3_col1.plotly_chart(fig_feat, use_container_width=True)
row3_col2.plotly_chart(fig_rev, use_container_width=True)

render_scenarios_table(scenarios)


# =========================
# METRICS DASHBOARD (design like reference image)
# =========================
rmse = float(trained_metrics.get("rmse", np.nan))
r2_score = float(trained_metrics.get("r2", np.nan))

mean_units = float(df["Units Sold"].mean())
mean_price = float(df["Price ($)"].mean())
mean_revenue = float(df["Revenue ($)"].mean())

demand_delta_pct = ((predicted_demand - mean_units) / mean_units * 100) if mean_units else 0.0
price_delta_pct = ((optimal_price - mean_price) / mean_price * 100) if mean_price else 0.0
revenue_delta_pct = ((revenue_projection - mean_revenue) / mean_revenue * 100) if mean_revenue else 0.0

def delta_badge_html(delta: float) -> str:
    cls = "negative" if delta < 0 else ""
    arrow = "▼" if delta < 0 else "▲"
    return f'<div class="metric-delta {cls}"><span class="metric-arrow">{arrow}</span> {delta:+.2f}%</div>'

performance_label = "Accurate Forecasts" if r2_score >= 0.75 else "Needs Calibration"

metrics_html = textwrap.dedent(
    f"""
    <div class="metrics-showcase">
      <h2 class="metrics-showcase-title">Metrics Dashboard</h2>
      <div class="metrics-showcase-divider"></div>
      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-box-head blue">Predicted Demand</div>
          <div class="metric-box-body">
            <p class="metric-main-value">{predicted_demand:,.0f}</p>
            {delta_badge_html(demand_delta_pct)}
          </div>
        </div>
        <div class="metric-box">
          <div class="metric-box-head orange">Optimal Price</div>
          <div class="metric-box-body">
            <p class="metric-main-value">${optimal_price:,.2f}</p>
            {delta_badge_html(price_delta_pct)}
          </div>
        </div>
        <div class="metric-box">
          <div class="metric-box-head green">Revenue Projection</div>
          <div class="metric-box-body">
            <p class="metric-main-value green">${revenue_projection:,.0f}</p>
            {delta_badge_html(revenue_delta_pct)}
          </div>
        </div>
        <div class="metric-box">
          <div class="metric-box-head sky">Model Performance</div>
          <div class="metric-box-body">
            <div class="perf-row">
              <div><strong>RMSE:</strong> <span class="big">{rmse:,.1f}</span></div>
              <div><strong>R² Score:</strong> <span class="big">{r2_score:.2f}</span></div>
            </div>
            <div class="perf-divider"></div>
            <div class="perf-status">
              <span class="perf-check">✓</span>
              <span>{performance_label}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
).strip()

st.markdown(metrics_html, unsafe_allow_html=True)
