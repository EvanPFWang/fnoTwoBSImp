
import os
import numpy as np
import streamlit as st
import plotly.express as px
from decimal import Decimal
from bs_pricing import black_scholes_call_put, build_grids, pnl_surface
from db import create_engine_from_url, create_all, insert_calculation

st.set_page_config(page_title="Black–Scholes Heatmap & PnL", layout="wide")
st.title("Black–Scholes (European) – Heatmap & PnL")

with st.sidebar:
    st.header("Database")
    default_url = os.environ.get("MYSQL_URL", "mysql+pymysql://user:pass@localhost:3306/Evan_dev_db")
    mysql_url = st.text_input("SQLAlchemy URL", value=default_url, help="e.g., mysql+pymysql://user:pass@host:3306/Evan_dev_db")
    auto_create = st.checkbox("Create tables if not exist", value=True)
    st.caption("Your credentials are only used by this app at runtime.")

st.subheader("Base Inputs")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    S0 = st.number_input("Spot S", min_value=0.0, value=100.0, step=1.0)
with col2:
    K = st.number_input("Strike K", min_value=0.0, value=100.0, step=1.0)
with col3:
    r = st.number_input("Rate r (dec.)", value=0.05, step=0.005, format="%.6f")
with col4:
    sigma = st.number_input("Vol σ (dec.)", min_value=0.0, value=0.2, step=0.01, format="%.6f")
with col5:
    T = st.number_input("Time to Expiry T (years)", min_value=0.0, value=1.0, step=0.25, format="%.6f")

st.subheader("Heatmap Parameters")
colA, colB, colC = st.columns([1,1,1])
with colA:
    min_S = st.number_input("Min Spot", min_value=0.0, value=50.0, step=1.0)
    max_S = st.number_input("Max Spot", min_value=0.0, value=150.0, step=1.0)
    nS = st.slider("Grid N (Spot)", min_value=10, max_value=200, value=50, step=5)
with colB:
    min_vol = st.number_input("Min Vol (dec.)", min_value=0.0, value=0.05, step=0.005, format="%.6f")
    max_vol = st.number_input("Max Vol (dec.)", min_value=0.0, value=0.50, step=0.01, format="%.6f")
    nV = st.slider("Grid N (Vol)", min_value=10, max_value=200, value=50, step=5)
with colC:
    inst = st.selectbox("Instrument for top heatmap", options=["Call", "Put"], index=0)
    show_values = st.checkbox("Annotate cells with values", value=False)

st.subheader("Your Trade (optional)")
colp1, colp2 = st.columns(2)
with colp1:
    purchase_call = st.number_input("Purchase Price (Call)", min_value=0.0, value=5.0, step=0.1, format="%.6f")
with colp2:
    purchase_put  = st.number_input("Purchase Price (Put)",  min_value=0.0, value=5.0, step=0.1, format="%.6f")

go = st.button("Calculate")

if go:
    #build grids and compute prices
    SS, VV, S_vec, V_vec = build_grids(min_S, max_S, min_vol, max_vol, nS, nV)
    call_surf, put_surf = black_scholes_call_put(SS, K, r, VV, T)

    #main heatmap (prices)
    main_surf = call_surf if inst == "Call" else put_surf
    fig = px.imshow(
        main_surf,
        x=S_vec, y=V_vec,
        origin="lower",
        aspect="auto",
        labels=dict(x="Spot (S)", y="Volatility (σ)", color="Price"),
        color_continuous_scale="RdYlGn"
    )
    if show_values:
        #annotate values (may be heavy for large grids)
        fig.update_traces(text=np.round(main_surf, 3), texttemplate="%{text}")
    st.plotly_chart(fig, use_container_width=True)

    #P&L heatmaps below
    st.subheader("P&L Surfaces")
    c1, c2 = st.columns(2)
    pnl_call = pnl_surface(call_surf, purchase_call)
    pnl_put  = pnl_surface(put_surf, purchase_put)

    fig_c = px.imshow(
        pnl_call, x=S_vec, y=V_vec, origin="lower", aspect="auto",
        labels=dict(x="Spot (S)", y="Volatility (σ)", color="PnL Call"),
        color_continuous_scale="RdYlGn"
    )
    fig_p = px.imshow(
        pnl_put, x=S_vec, y=V_vec, origin="lower", aspect="auto",
        labels=dict(x="Spot (S)", y="Volatility (σ)", color="PnL Put"),
        color_continuous_scale="RdYlGn"
    )
    if show_values:
        fig_c.update_traces(text=np.round(pnl_call, 3), texttemplate="%{text}")
        fig_p.update_traces(text=np.round(pnl_put, 3),  texttemplate="%{text}")
    c1.plotly_chart(fig_c, use_container_width=True)
    c2.plotly_chart(fig_p, use_container_width=True)

    #save to MySQL
    try:
        engine = create_engine_from_url(mysql_url)
        if auto_create:
            create_all(engine)
        inputs = dict(
            StockPrice=S0, StrikePrice=K, InterestRate=r, Volatility=sigma, TimeToExpiry=T,
            PurchaseCallPrice=purchase_call, PurchasePutPrice=purchase_put,
            MinSpot=min_S, MaxSpot=max_S, MinVol=min_vol, MaxVol=max_vol,
            GridNSpot=nS, GridNVol=nV
        )
        calc_id, n_rows = insert_calculation(engine, inputs, S_vec, V_vec, call_surf, put_surf)
        st.success(f"Saved CalculationId={calc_id}, rows inserted={n_rows}.")
    except Exception as e:
        st.error(f"DB save failed: {e}")

with st.expander("Advanced (PDE/Parareal) – optional hooks"):
    st.caption("If local modules exist (e.g., parareal, forward_euler), this app can show they are importable.")
    try:
        import importlib
        mods = []
        for name in ["black_scholes_exact", "forward_euler", "iterate_solution", "parareal"]:
            try:
                importlib.import_module(name)
                mods.append(name)
            except Exception:
                pass
        if mods:
            st.info("Detected modules: " + ", ".join(mods))
        else:
            st.write("No optional modules found on PYTHONPATH.")
    except Exception as e:
        st.write("Module check error:", e)
