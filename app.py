"""
Nassau Candy — Factory Reallocation & Shipping Optimization Dashboard
Author: Hemant Sharma | Internship: Unified Mentor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy Dashboard",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFB; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1B4F72 0%, #2E86C1 100%); }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: white !important; }
    .kpi-card { background: white; border-radius: 12px; padding: 20px 16px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 5px solid #2E86C1; margin-bottom: 10px; }
    .kpi-card-green { border-left-color: #27AE60 !important; }
    .kpi-card-amber { border-left-color: #E67E22 !important; }
    .kpi-card-red   { border-left-color: #C0392B !important; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #1B2631; margin: 6px 0 2px; }
    .kpi-label { font-size: 12px; color: #7F8C8D; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .section-header { font-size: 16px; font-weight: 700; color: #1B4F72; border-bottom: 2px solid #2E86C1; padding-bottom: 6px; margin: 18px 0 12px; }
    .author-badge { background: rgba(255,255,255,0.15); border-radius: 8px; padding: 10px 14px; margin-top: 20px; font-size: 13px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING & PIPELINE ──────────────────────────────────
@st.cache_data
def load_data():
    # 1. Load Sheets
    xls = pd.ExcelFile("Nassau Candy DistributorEXACTDATAFILE.xlsx")
    df = pd.read_excel(xls, sheet_name="Nassau Candy Distributor")
    factories = pd.read_excel(xls, sheet_name="FACTORY locations")
    divisions = pd.read_excel(xls, sheet_name="Divisions")

    # 2. Date Cleaning & Shipping
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce") - pd.DateOffset(years=2)
    df["Shipping_Duration_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days.clip(lower=0)
    
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Year"] = df["Order Date"].dt.year.astype(str)
    df["Season"] = df["Order Date"].dt.month.map({
        12:"Winter",1:"Winter",2:"Winter", 3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer", 9:"Fall",10:"Fall",11:"Fall"
    })
    
    # 3. Margin & KMeans Clustering
    df["Margin_Pct"] = np.where(df["Sales"] > 0, df["Gross Profit"] / df["Sales"], 0)
    
    features = df[["Sales", "Shipping_Duration_Days", "Margin_Pct"]].fillna(0)
    X = StandardScaler().fit_transform(features)
    df["Cluster_Raw"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    
    cluster_summary = df.groupby("Cluster_Raw")[["Sales", "Shipping_Duration_Days"]].mean()
    high_risk_id = cluster_summary["Shipping_Duration_Days"].idxmax()
    premium_id = cluster_summary["Sales"].idxmax()
    
    def label_cluster(c):
        if c == high_risk_id: return "High-Risk / Delayed"
        elif c == premium_id and c != high_risk_id: return "Premium / High-Value"
        return "Standard / Baseline"
    df["Cluster_Persona"] = df["Cluster_Raw"].apply(label_cluster)

    # 4. Factory Mapping & Distances
    df = df.merge(divisions[["Product Name", "Factory"]], on="Product Name", how="left").rename(columns={"Factory": "Current_Factory"}).fillna({"Current_Factory":"Unknown"})

    STATE_COORDS = {
        "Alabama": (32.8, -86.8), "California": (36.1, -119.7), "Florida": (27.8, -81.7),
        "Illinois": (40.3, -89.0), "New York": (42.2, -74.9), "Texas": (31.1, -97.6),
        "Washington": (47.4, -121.5), "Ohio": (40.4, -82.8), "Pennsylvania": (40.6, -77.2)
    }
    df["State_Lat"] = df["State/Province"].map(lambda s: STATE_COORDS.get(s, (39.8, -98.5))[0])
    df["State_Lon"] = df["State/Province"].map(lambda s: STATE_COORDS.get(s, (39.8, -98.5))[1])

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2
        return 2 * 6371 * np.arcsin(np.sqrt(a))
    
    factories.columns = factories.columns.str.strip()
    dist_cols = []
    for _, frow in factories.iterrows():
        col_name = f"Dist_{frow['Factory']}"
        df[col_name] = haversine(df["State_Lat"], df["State_Lon"], frow["Latitude"], frow["Longitude"])
        dist_cols.append(col_name)
    
    df["Current_Distance_km"] = df.apply(lambda r: r[f"Dist_{r['Current_Factory']}"] if f"Dist_{r['Current_Factory']}" in r else r[dist_cols].mean(), axis=1)
    df["Optimal_Factory"] = df[dist_cols].idxmin(axis=1).str.replace("Dist_", "", regex=False)
    df["Optimal_Distance_km"] = df[dist_cols].min(axis=1)

    df["Distance_Saved_km"] = df["Current_Distance_km"] - df["Optimal_Distance_km"]
    df["Needs_Reallocation"] = df["Current_Factory"] != df["Optimal_Factory"]
    df["Cost_Savings"] = df["Distance_Saved_km"] * 0.01
    
    df["Current_Lead_Days"] = 1 + np.ceil(df["Current_Distance_km"] / 600)
    df["Optimal_Lead_Days"] = 1 + np.ceil(df["Optimal_Distance_km"] / 600)
    df["Lead_Time_Reduction_Days"] = df["Current_Lead_Days"] - df["Optimal_Lead_Days"]

    # 5. Ship Recommendations
    def ship_recommendation(row):
        if row["Cluster_Persona"] == "High-Risk / Delayed" and row["Ship Mode"] == "Standard Class": return "Upgrade to First Class (Premium Service)"
        if row["Ship Mode"] in ("First Class", "Same Day") and row["Margin_Pct"] < 0.30: return "Downgrade to Standard (Cost Saving)"
        return "Optimal Configuration Maintained"
    df["Ship_Recommendation"] = df.apply(ship_recommendation, axis=1)
    
    def ship_financial_impact(row):
        if str(row["Ship_Recommendation"]).startswith("Upgrade"): return -row["Cost"] * 0.07
        if str(row["Ship_Recommendation"]).startswith("Downgrade"): return row["Cost"] * 0.05
        return 0.0
    
    df["Ship_Financial_Impact"] = df.apply(ship_financial_impact, axis=1)
    df["Total_Financial_Impact"] = df["Ship_Financial_Impact"] + df["Cost_Savings"]
    df["Has_Recommendation"] = (df["Ship_Recommendation"] != "Optimal Configuration Maintained") | df["Needs_Reallocation"]

    return df

df = load_data()

COLORS = {"navy": "#1B4F72", "blue": "#2E86C1", "green": "#27AE60", "amber": "#E67E22", "red": "#C0392B", "light": "#85C1E9"}
PALETTE = [COLORS["navy"], COLORS["blue"], COLORS["green"], COLORS["amber"], COLORS["red"], COLORS["light"]]

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍬 Nassau Candy")
    st.markdown("**Factory Reallocation &**  \n**Shipping Optimization**")
    st.markdown("---")
    st.markdown("### 🔍 Filters")

    region = st.multiselect("Region", sorted(df["Region"].dropna().unique()), default=sorted(df["Region"].dropna().unique()))
    season = st.multiselect("Season", ["Fall","Spring","Summer","Winter"], default=["Fall","Spring","Summer","Winter"])
    ship_mode = st.multiselect("Ship Mode", sorted(df["Ship Mode"].dropna().unique()), default=sorted(df["Ship Mode"].dropna().unique()))
    cluster = st.multiselect("Order Segment", sorted(df["Cluster_Persona"].dropna().unique()), default=sorted(df["Cluster_Persona"].dropna().unique()))

    st.markdown("---")
    st.markdown("### 📊 Navigation")
    page = st.radio("", ["🏠 Overview", "🏭 Factory Analysis", "🚚 Shipping & Orders", "📦 Product Insights", "⚠️ Risk & Impact"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""<div class="author-badge">👤 <b>Hemant Sharma</b><br>🏢 Unified Mentor Internship<br>📅 June 2026</div>""", unsafe_allow_html=True)

# ── FILTER DATA ──────────────────────────────────────────────
fdf = df[df["Region"].isin(region) & df["Season"].isin(season) & df["Ship Mode"].isin(ship_mode) & df["Cluster_Persona"].isin(cluster)].copy()

total_sales = fdf["Sales"].sum()
total_gp    = fdf["Gross Profit"].sum()
margin      = total_gp / total_sales if total_sales else 0
realloc_pct = fdf["Needs_Reallocation"].mean() if not fdf.empty else 0
coverage    = fdf["Has_Recommendation"].mean() if not fdf.empty else 0

# ── TITLE BAR ────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(90deg,#1B4F72,#2E86C1); padding:18px 24px;border-radius:12px;margin-bottom:20px; display:flex;justify-content:space-between;align-items:center;">
  <div>
    <span style="color:white;font-size:22px;font-weight:700;">🍬 Nassau Candy — Factory Reallocation & Shipping Optimization</span><br>
    <span style="color:#D4E6F1;font-size:13px;">Hemant Sharma &nbsp;|&nbsp; Unified Mentor Internship &nbsp;|&nbsp; {len(fdf):,} of {len(df):,} orders shown</span>
  </div>
  <div style="color:#D4E6F1;font-size:12px;text-align:right;">Data: Jan 2024 – Dec 2025<br>Last updated: Dec 2025</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    kpis = [
        (c1, "Total Sales", f"${total_sales/1000:.1f}K", "blue"),
        (c2, "Gross Profit", f"${total_gp/1000:.1f}K", "green"),
        (c3, "Gross Margin", f"{margin:.1%}", "green"),
        (c4, "Orders", f"{len(fdf):,}", "blue"),
        (c5, "Realloc. Needed", f"{realloc_pct:.1%}", "amber"),
        (c6, "Rec. Coverage", f"{coverage:.1%}", "blue"),
        (c7, "Cost Savings", f"${fdf['Cost_Savings'].sum():,.0f}", "green"),
    ]
    for col, label, value, color in kpis:
        col.markdown(f"""<div class="kpi-card kpi-card-{color}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">📈 Gross Profit by Month</div>', unsafe_allow_html=True)
        monthly = fdf.groupby("Month").agg(GP=("Gross Profit","sum"), Sales=("Sales","sum")).reset_index().sort_values("Month")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["GP"], mode="lines+markers", name="Gross Profit", line=dict(color=COLORS["blue"], width=2.5), fill="tozeroy", fillcolor="rgba(46,134,193,0.1)", marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Sales"], mode="lines", name="Sales", line=dict(color=COLORS["amber"], width=1.5, dash="dot")))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h", y=1.1), plot_bgcolor="white", paper_bgcolor="white", xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=9)), yaxis=dict(showgrid=True, gridcolor="#F0F0F0", tickprefix="$", tickformat=".0f"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🍫 Sales by Division</div>', unsafe_allow_html=True)
        div = fdf.groupby("Division").agg(Sales=("Sales","sum"), GP=("Gross Profit","sum")).reset_index()
        fig2 = px.pie(div, values="Sales", names="Division", color_discrete_sequence=PALETTE, hole=0.45)
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), showlegend=False, paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">🗺️ Sales by Region</div>', unsafe_allow_html=True)
        reg = fdf.groupby("Region").agg(Sales=("Sales","sum"), GP=("Gross Profit","sum"), Orders=("Order ID","count")).reset_index().sort_values("GP", ascending=True)
        fig3 = px.bar(reg, x="GP", y="Region", orientation="h", color="Region", color_discrete_sequence=PALETTE, labels={"GP":"Gross Profit ($)"})
        fig3.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), showlegend=False, plot_bgcolor="white", paper_bgcolor="white", xaxis=dict(showgrid=True, gridcolor="#F0F0F0", tickprefix="$"))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">🗓️ Performance by Season</div>', unsafe_allow_html=True)
        seas = fdf.groupby("Season").agg(GP=("Gross Profit","sum"), Orders=("Order ID","count")).reset_index().sort_values("GP", ascending=False)
        fig4 = px.bar(seas, x="Season", y="GP", color="Season", color_discrete_sequence=PALETTE, labels={"GP":"Gross Profit ($)"}, text="GP")
        fig4.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig4.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), showlegend=False, plot_bgcolor="white", paper_bgcolor="white", yaxis=dict(showgrid=True, gridcolor="#F0F0F0"))
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: FACTORY ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🏭 Factory Analysis":
    st.markdown('<div class="section-header">🏭 Factory Reallocation Analysis</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Orders Needing Reallocation", f"{fdf['Needs_Reallocation'].sum():,}", f"{fdf['Needs_Reallocation'].mean():.1%} of total")
    m2.metric("Total Distance Saved", f"{fdf['Distance_Saved_km'].sum():,.0f} km")
    m3.metric("Total Cost Savings", f"${fdf['Cost_Savings'].sum():,.0f}")
    m4.metric("Avg Lead Time Reduction", f"{fdf['Lead_Time_Reduction_Days'].mean():.1f} days")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Current vs Optimal Factory</div>', unsafe_allow_html=True)
        flow = fdf.groupby(["Current_Factory","Optimal_Factory"]).size().reset_index(name="Orders")
        flow = flow[flow["Orders"] > 0]
        fig5 = px.bar(flow, x="Orders", y="Optimal_Factory", color="Current_Factory", orientation="h", barmode="stack", color_discrete_sequence=PALETTE, labels={"Optimal_Factory":"Optimal Factory","Orders":"Order Count"})
        fig5.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="white", paper_bgcolor="white", legend=dict(title="Current Factory", orientation="h", y=-0.3))
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Cost Savings by Optimal Factory</div>', unsafe_allow_html=True)
        fs = fdf.groupby("Optimal_Factory").agg(Cost_Savings=("Cost_Savings","sum"), Distance_Saved=("Distance_Saved_km","sum"), Orders=("Order ID","count")).reset_index().sort_values("Cost_Savings", ascending=True)
        fig6 = px.bar(fs, x="Cost_Savings", y="Optimal_Factory", orientation="h", color="Cost_Savings", color_continuous_scale=["#D4E6F1","#1B4F72"], labels={"Cost_Savings":"Cost Savings ($)","Optimal_Factory":"Factory"})
        fig6.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="section-header">📋 Per-Product Factory Assignment</div>', unsafe_allow_html=True)
    prod_factory = fdf.groupby(["Product Name","Current_Factory","Optimal_Factory"]).agg(Orders=("Order ID","count"), Avg_Distance_Current=("Current_Distance_km","mean"), Avg_Distance_Optimal=("Optimal_Distance_km","mean"), Total_Savings=("Cost_Savings","sum")).round(2).reset_index()
    prod_factory["Needs Reallocation"] = (prod_factory["Current_Factory"] != prod_factory["Optimal_Factory"]).map({True:"⚠️ Yes", False:"✅ No"})
    st.dataframe(prod_factory, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE: SHIPPING & ORDERS
# ══════════════════════════════════════════════════════════════
elif page == "🚚 Shipping & Orders":
    st.markdown('<div class="section-header">🚚 Shipping Mode Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        ship = fdf.groupby("Ship Mode").agg(Sales=("Sales","sum"), GP=("Gross Profit","sum"), Orders=("Order ID","count"), Ship_Impact=("Ship_Financial_Impact","sum")).reset_index()
        fig7 = px.bar(ship, x="Ship Mode", y=["GP","Ship_Impact"], barmode="group", color_discrete_map={"GP": COLORS["blue"], "Ship_Impact": COLORS["amber"]}, labels={"value":"Amount ($)","variable":"Metric"})
        fig7.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), plot_bgcolor="white", paper_bgcolor="white", title="Gross Profit vs Shipping Impact by Mode", title_font_size=13)
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        rec = fdf.groupby("Ship_Recommendation").agg(Orders=("Order ID","count"), GP=("Gross Profit","sum"), Impact=("Ship_Financial_Impact","sum")).reset_index()
        fig8 = px.bar(rec, x="GP", y="Ship_Recommendation", orientation="h", color="Ship_Recommendation", color_discrete_sequence=PALETTE, labels={"Ship_Recommendation":"Recommendation","GP":"Gross Profit ($)"})
        fig8.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), showlegend=False, plot_bgcolor="white", paper_bgcolor="white", title="Orders by Shipping Recommendation", title_font_size=13)
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="section-header">👥 Order Behaviour Profiles (KMeans Segments)</div>', unsafe_allow_html=True)
    col3, col4 = st.columns([2, 3])

    with col3:
        seg = fdf.groupby("Cluster_Persona").agg(Total_Orders=("Order ID","count"), Avg_Sales=("Sales","mean"), Total_GP=("Gross Profit","sum"), Avg_Shipping_Days=("Shipping_Duration_Days","mean"), Overall_Margin=("Margin_Pct","mean")).round(2).reset_index()
        seg.columns = ["Segment","Orders","Avg Sales","Total GP","Avg Ship Days","Margin"]
        seg["Margin"] = (seg["Margin"]*100).round(1).astype(str) + "%"
        seg["Avg Sales"] = seg["Avg Sales"].apply(lambda x: f"${x:.2f}")
        seg["Total GP"] = seg["Total GP"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(seg, use_container_width=True, hide_index=True)

    with col4:
        fig9 = px.scatter(fdf.sample(min(3000, len(fdf)), random_state=42), x="Sales", y="Margin_Pct", color="Cluster_Persona", size="Shipping_Duration_Days", color_discrete_sequence=PALETTE, labels={"Sales":"Order Sales ($)","Margin_Pct":"Margin %","Cluster_Persona":"Segment"}, opacity=0.6)
        fig9.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=-0.3))
        fig9.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig9, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: PRODUCT INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "📦 Product Insights":
    st.markdown('<div class="section-header">📦 Product Portfolio Analysis</div>', unsafe_allow_html=True)
    prod = fdf.groupby("Product Name").agg(Sales=("Sales","sum"), GP=("Gross Profit","sum"), Cost=("Cost","sum"), Orders=("Order ID","count"), Units=("Units","sum")).reset_index()
    prod["Margin %"] = (prod["GP"] / prod["Sales"] * 100).round(2)
    prod["Margin Band"] = prod["Margin %"].apply(lambda x: "🔴 Critical (<20%)" if x < 20 else ("🟡 Low (20-50%)" if x < 50 else ("🟢 Acceptable (50-65%)" if x < 65 else "✅ Strong (>65%)")))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Product Margin Scatter** (bubble = order count)")
        fig10 = px.scatter(prod, x="Sales", y="Margin %", size="Orders", color="Margin Band", hover_name="Product Name", color_discrete_map={"🔴 Critical (<20%)": COLORS["red"], "🟡 Low (20-50%)": COLORS["amber"], "🟢 Acceptable (50-65%)": COLORS["light"], "✅ Strong (>65%)": COLORS["green"]}, labels={"Sales":"Total Sales ($)"}, size_max=25)
        fig10.add_hline(y=20, line_dash="dash", line_color=COLORS["red"], annotation_text="Low Margin Threshold (20%)", annotation_position="top right")
        fig10.add_hline(y=65, line_dash="dash", line_color=COLORS["green"], annotation_text="Strong Margin (65%)", annotation_position="bottom right")
        fig10.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=-0.3, title=""))
        st.plotly_chart(fig10, use_container_width=True)

    with col2:
        st.markdown("**Gross Profit by Product**")
        prod_sorted = prod.sort_values("GP", ascending=True)
        fig11 = px.bar(prod_sorted, x="GP", y="Product Name", orientation="h", color="Margin %", color_continuous_scale=["#C0392B","#E67E22","#27AE60"], labels={"GP":"Gross Profit ($)","Product Name":""}, range_color=[0,80])
        fig11.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="white", paper_bgcolor="white", coloraxis_colorbar=dict(title="Margin %"))
        st.plotly_chart(fig11, use_container_width=True)

    st.markdown('<div class="section-header">📋 Full Product Table</div>', unsafe_allow_html=True)
    display_prod = prod[["Product Name","Sales","GP","Cost","Orders","Units","Margin %","Margin Band"]].copy()
    display_prod["Sales"] = display_prod["Sales"].apply(lambda x: f"${x:,.2f}")
    display_prod["GP"]    = display_prod["GP"].apply(lambda x: f"${x:,.2f}")
    display_prod["Cost"]  = display_prod["Cost"].apply(lambda x: f"${x:,.2f}")
    display_prod["Margin %"] = display_prod["Margin %"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(display_prod.sort_values("Margin %", ascending=True), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE: RISK & IMPACT
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Risk & Impact":
    st.markdown('<div class="section-header">⚠️ Risk & Impact Analysis</div>', unsafe_allow_html=True)
    total_impact = fdf["Total_Financial_Impact"].sum()
    if total_impact >= 0: st.success(f"✅ Net financial impact for current selection: **${total_impact:,.0f}** (positive)")
    else: st.error(f"🔴 Net financial impact for current selection: **${total_impact:,.0f}** (negative)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Risk & Impact Quadrant**")
        sample = fdf.sample(min(3000, len(fdf)), random_state=42)
        fig12 = px.scatter(sample, x="Cost_Savings", y="Lead_Time_Reduction_Days", color="Optimal_Factory", opacity=0.65, color_discrete_sequence=PALETTE, labels={"Cost_Savings":"Cost Savings ($)", "Lead_Time_Reduction_Days":"Lead Time Reduction (days)", "Optimal_Factory":"Factory"}, size_max=8)
        fig12.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig12.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
        fig12.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig12, use_container_width=True)

    with col2:
        st.markdown("**Segment-Level Stability**")
        stab = fdf.groupby("Cluster_Persona")["Total_Financial_Impact"].agg(["mean","std","count"]).round(2).reset_index()
        stab.columns = ["Segment","Avg Impact ($)","Std Dev ($)","Orders"]
        st.dataframe(stab, use_container_width=True, hide_index=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#7F8C8D;font-size:12px;padding:10px 0;">🍬 Nassau Candy — Factory Reallocation & Shipping Optimization Dashboard &nbsp;|&nbsp; Built by <b>Hemant Sharma</b> &nbsp;|&nbsp; Unified Mentor Internship &nbsp;|&nbsp; June 2026</div>""", unsafe_allow_html=True)
