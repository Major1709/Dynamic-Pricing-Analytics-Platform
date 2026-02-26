import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Analyse de donnees", layout="wide")


@st.cache_data
def load_data(path: str = "pricing_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def apply_theme() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: #f3f5f9;
            color: #243b53;
          }

          .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #163d6d !important;
          }

          .block-container {
            max-width: 1500px;
            padding-top: 1.0rem;
            padding-bottom: 1.2rem;
          }

          .ana-card {
            background: #ffffff;
            border: 1px solid #e2e8f1;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
          }

          .ana-section-title {
            color: #163d6d;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 0.55rem 0;
          }

          .ana-subtext {
            color: #5b6f86;
            font-size: 0.9rem;
            margin: 0;
          }

          [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f1;
            border-radius: 10px;
            padding: 0.45rem 0.7rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
          }

          [data-testid="stMetricLabel"] p {
            color: #5b6f86 !important;
            font-weight: 600;
          }

          [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            color: #163d6d !important;
          }

          [data-testid="stPlotlyChart"] {
            background: #ffffff;
            border: 1px solid #e2e8f1;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            padding: 0.25rem 0.25rem 0 0.25rem;
          }

          div[data-testid="stPlotlyChart"] .js-plotly-plot text {
            fill: #243b53 !important;
          }

          .table-wrap {
            background: #ffffff;
            border: 1px solid #e2e8f1;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            padding: 0.5rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig, height: int = 320):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=26, r=18, t=48, b=28),
        font=dict(color="#243b53"),
        title=dict(font=dict(color="#163d6d", size=16), x=0.02, xanchor="left"),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", borderwidth=0),
    )
    fig.update_xaxes(linecolor="#dbe5f0", tickfont=dict(color="#243b53"))
    fig.update_yaxes(linecolor="#dbe5f0", gridcolor="#e9eef5", tickfont=dict(color="#243b53"))
    return fig


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Price ($)", "Competitor Price ($)", "Revenue ($)"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )
    if "Units Sold" in out.columns:
        out["Units Sold"] = pd.to_numeric(out["Units Sold"], errors="coerce").map(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )
    if "Date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["Date"]):
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


apply_theme()
df = load_data()

st.title("Analyse de donnees")
st.caption("Exploration du dataset de pricing: filtres, statistiques, distributions et correlations.")

if df.empty:
    st.error("Le fichier pricing_dataset.csv est vide ou introuvable.")
    st.stop()

# Prepare options
df_work = df.copy()
if "Date" in df_work.columns and pd.api.types.is_datetime64_any_dtype(df_work["Date"]):
    min_date = df_work["Date"].min()
    max_date = df_work["Date"].max()
else:
    min_date = max_date = None

with st.sidebar:
    st.header("Filtres")

    if min_date is not None and max_date is not None and pd.notna(min_date) and pd.notna(max_date):
        date_range = st.date_input(
            "Periode",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        else:
            start_date = end_date = None
    else:
        start_date = end_date = None

    season_vals = sorted(df_work["Season"].dropna().astype(str).unique()) if "Season" in df_work.columns else []
    season_sel = st.multiselect("Season", season_vals, default=season_vals)

    promo_vals = sorted(df_work["Promo"].dropna().astype(str).unique()) if "Promo" in df_work.columns else []
    promo_sel = st.multiselect("Promo", promo_vals, default=promo_vals)

    tod_vals = sorted(df_work["Time of Day"].dropna().astype(str).unique()) if "Time of Day" in df_work.columns else []
    tod_sel = st.multiselect("Time of Day", tod_vals, default=tod_vals)

    if "Price ($)" in df_work.columns:
        pmin = float(pd.to_numeric(df_work["Price ($)"], errors="coerce").min())
        pmax = float(pd.to_numeric(df_work["Price ($)"], errors="coerce").max())
        price_range = st.slider("Price range ($)", min_value=float(round(pmin, 2)), max_value=float(round(pmax, 2)), value=(float(round(pmin, 2)), float(round(pmax, 2))))
    else:
        price_range = None

    search = st.text_input("Recherche texte", placeholder="Ex: Summer, Morning, 39.99")

    max_rows = st.slider("Lignes dans le tableau", min_value=10, max_value=200, value=30, step=10)

# Apply filters
filtered = df_work.copy()

if start_date is not None and end_date is not None and "Date" in filtered.columns:
    filtered = filtered[(filtered["Date"] >= start_date) & (filtered["Date"] <= end_date)]

if season_sel and "Season" in filtered.columns:
    filtered = filtered[filtered["Season"].astype(str).isin(season_sel)]
elif "Season" in filtered.columns and season_vals:
    filtered = filtered.iloc[0:0]

if promo_sel and "Promo" in filtered.columns:
    filtered = filtered[filtered["Promo"].astype(str).isin(promo_sel)]
elif "Promo" in filtered.columns and promo_vals:
    filtered = filtered.iloc[0:0]

if tod_sel and "Time of Day" in filtered.columns:
    filtered = filtered[filtered["Time of Day"].astype(str).isin(tod_sel)]
elif "Time of Day" in filtered.columns and tod_vals:
    filtered = filtered.iloc[0:0]

if price_range is not None and "Price ($)" in filtered.columns:
    price_num = pd.to_numeric(filtered["Price ($)"], errors="coerce")
    filtered = filtered[(price_num >= price_range[0]) & (price_num <= price_range[1])]

if search.strip():
    q = search.strip().lower()
    mask = pd.Series(False, index=filtered.index)
    for col in filtered.columns:
        mask = mask | filtered[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
    filtered = filtered[mask]

row_count = len(filtered)

# KPIs
units_mean = pd.to_numeric(filtered.get("Units Sold"), errors="coerce").mean() if "Units Sold" in filtered.columns else None
price_mean = pd.to_numeric(filtered.get("Price ($)"), errors="coerce").mean() if "Price ($)" in filtered.columns else None
revenue_sum = pd.to_numeric(filtered.get("Revenue ($)"), errors="coerce").sum() if "Revenue ($)" in filtered.columns else None
promo_rate = None
if "Promo" in filtered.columns and row_count > 0:
    promo_rate = (filtered["Promo"].astype(str).str.lower() == "yes").mean() * 100

st.markdown('<div class="ana-card"><div class="ana-section-title">Vue d\'ensemble</div><p class="ana-subtext">Statistiques globales sur les donnees filtrees.</p></div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lignes", f"{row_count:,}")
k2.metric("Prix moyen", f"${price_mean:,.2f}" if pd.notna(price_mean) else "-")
k3.metric("Ventes moyennes", f"{int(round(units_mean)):,}" if pd.notna(units_mean) else "-")
k4.metric("Revenue total", f"${revenue_sum:,.0f}" if pd.notna(revenue_sum) else "-")

k5, k6 = st.columns(2)
k5.metric("Taux Promo", f"{promo_rate:.1f}%" if promo_rate is not None else "-")
if "Date" in filtered.columns and row_count > 0 and pd.api.types.is_datetime64_any_dtype(filtered["Date"]):
    span = filtered["Date"].max() - filtered["Date"].min()
    k6.metric("Periode couverte", f"{span.days + 1} jours")
else:
    k6.metric("Periode couverte", "-")

if row_count == 0:
    st.warning("Aucune donnee apres filtrage. Ajustez les filtres dans la barre laterale.")
    st.stop()

# Charts
num_cols = [c for c in ["Price ($)", "Units Sold", "Competitor Price ($)", "Revenue ($)"] if c in filtered.columns]

col_a, col_b = st.columns(2)
with col_a:
    if {"Price ($)", "Units Sold"}.issubset(filtered.columns):
        fig = px.scatter(
            filtered.sample(min(len(filtered), 1500), random_state=42),
            x="Price ($)",
            y="Units Sold",
            color="Promo" if "Promo" in filtered.columns else None,
            title="Relation Prix vs Ventes",
            opacity=0.65,
        )
        style_plot(fig, 340)
        st.plotly_chart(fig, use_container_width=True)
with col_b:
    if "Revenue ($)" in filtered.columns:
        fig = px.histogram(filtered, x="Revenue ($)", nbins=35, title="Distribution du Revenue")
        fig.update_traces(marker_color="#4dbb99", marker_line_color="#ffffff", marker_line_width=0.4)
        style_plot(fig, 340)
        st.plotly_chart(fig, use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    if "Date" in filtered.columns and "Units Sold" in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered["Date"]):
        by_day = filtered.groupby(filtered["Date"].dt.date)["Units Sold"].sum().reset_index()
        by_day["Date"] = pd.to_datetime(by_day["Date"])
        fig = px.line(by_day, x="Date", y="Units Sold", title="Evolution des ventes par jour")
        fig.update_traces(line=dict(color="#2d6bd0", width=3))
        style_plot(fig, 340)
        st.plotly_chart(fig, use_container_width=True)
with col_d:
    if "Season" in filtered.columns and "Units Sold" in filtered.columns:
        by_season = filtered.groupby("Season", as_index=False)["Units Sold"].mean()
        fig = px.bar(by_season, x="Season", y="Units Sold", title="Ventes moyennes par saison")
        fig.update_traces(marker_color="#6ba4ea", marker_line_color="#2d6bd0", marker_line_width=1)
        style_plot(fig, 340)
        st.plotly_chart(fig, use_container_width=True)

col_e, col_f = st.columns(2)
with col_e:
    if "Time of Day" in filtered.columns and "Revenue ($)" in filtered.columns:
        by_tod = filtered.groupby("Time of Day", as_index=False)["Revenue ($)"].mean()
        fig = px.bar(by_tod, x="Time of Day", y="Revenue ($)", title="Revenue moyen par Time of Day")
        fig.update_traces(marker_color="#f08b23", marker_line_color="#d97706", marker_line_width=1)
        style_plot(fig, 320)
        st.plotly_chart(fig, use_container_width=True)
with col_f:
    if "Weekday" in filtered.columns and "Units Sold" in filtered.columns:
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_weekday = filtered.groupby("Weekday", as_index=False)["Units Sold"].mean()
        by_weekday["__ord"] = by_weekday["Weekday"].apply(lambda x: weekday_order.index(x) if x in weekday_order else 99)
        by_weekday = by_weekday.sort_values("__ord")
        fig = px.line(by_weekday, x="Weekday", y="Units Sold", markers=True, title="Ventes moyennes par jour de semaine")
        fig.update_traces(line=dict(color="#38a37f", width=3), marker=dict(size=8))
        style_plot(fig, 320)
        st.plotly_chart(fig, use_container_width=True)

# Correlation and stats
stats_col, corr_col = st.columns([1.05, 1.35])

with stats_col:
    st.markdown(
        '<div class="ana-card"><div class="ana-section-title">Statistiques descriptives</div><p class="ana-subtext">Resume des colonnes numeriques.</p></div>',
        unsafe_allow_html=True,
    )
    numeric = filtered.select_dtypes(include="number")
    if not numeric.empty:
        st.dataframe(numeric.describe().T.round(2), use_container_width=True)
    else:
        st.info("Pas de colonnes numeriques disponibles.")

with corr_col:
    st.markdown(
        '<div class="ana-card"><div class="ana-section-title">Correlation</div><p class="ana-subtext">Heatmap des correlations numeriques.</p></div>',
        unsafe_allow_html=True,
    )
    numeric = filtered.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        corr = numeric.corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Correlation Matrix",
        )
        style_plot(fig, 420)
        fig.update_layout(coloraxis_colorbar=dict(title="Corr"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Il faut au moins 2 colonnes numeriques pour la correlation.")

# Missing values
st.markdown(
    '<div class="ana-card"><div class="ana-section-title">Qualite des donnees</div><p class="ana-subtext">Valeurs manquantes par colonne (sur les donnees filtrees).</p></div>',
    unsafe_allow_html=True,
)
missing_df = pd.DataFrame(
    {
        "Colonne": filtered.columns,
        "Valeurs manquantes": [int(filtered[c].isna().sum()) for c in filtered.columns],
        "Pourcentage (%)": [round(filtered[c].isna().mean() * 100, 2) for c in filtered.columns],
    }
).sort_values(["Valeurs manquantes", "Colonne"], ascending=[False, True])
st.dataframe(missing_df, use_container_width=True, hide_index=True)

# Data preview
st.markdown(
    '<div class="ana-card"><div class="ana-section-title">Apercu des donnees filtrees</div><p class="ana-subtext">Tableau detaille avec mise en forme.</p></div>',
    unsafe_allow_html=True,
)
st.dataframe(format_table(filtered.head(max_rows)), use_container_width=True, hide_index=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Telecharger les donnees filtrees (CSV)",
    data=csv_bytes,
    file_name="analyse_donnees_filtrees.csv",
    mime="text/csv",
)
