
import os
import numpy as np
import streamlit as st
import plotly.express as px
from decimal import Decimal
from bs_pricing.py import black_scholes_call_put, build_grids, pnl_surface
from db import create_engine_from_url, create_all, insert_calculation
import datetime as dt



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








#Custom CSS for enhanced appearance
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 1rem;
    }

    /* Call option styling */
    .call-option {
        background-color: #90ee90;
        padding: 10px;
        border-radius: 5px;
        color: black;
    }

    /* Put option styling */
    .put-option {
        background-color: #ffcccb;
        padding: 10px;
        border-radius: 5px;
        color: black;
    }

    /* Calculate button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #45a049;
    }

    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

#Title and description
st.title("Black-Scholes Option Pricing Model")
st.markdown("*Interactive option pricing with heatmaps, P&L analysis, and Greeks calculations*")

#Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    #Base Input Parameters
    st.subheader("Base Parameters")

    col1, col2 = st.columns(2)
    with col1:
        S0 = st.number_input(
            "Stock Price (S₀)",
            min_value=0.01,
            value=100.0,
            step=0.01,
            format="%.2f",
            help="Current price of the underlying stock"
        )
    with col2:
        K = st.number_input(
            "Strike Price (K)",
            min_value=0.01,
            value=100.0,
            step=0.01,
            format="%.2f",
            help="Strike price of the option"
        )

    col3, col4 = st.columns(2)
    with col3:
        T = st.number_input(
            "Time to Expiry (Years)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.25,
            format="%.2f",
            help="Time until option expiration in years"
        )
    with col4:
        r = st.number_input(
            "Risk-Free Rate",
            min_value=-0.1,
            max_value=1.0,
            value=0.05,
            step=0.001,
            format="%.3f",
            help="Annual risk-free interest rate"
        )

    #Volatility with both slider and input
    sigma = st.slider(
        "Volatility (σ)",
        min_value=0.01,
        max_value=2.0,
        value=0.2,
        step=0.01,
        format="%.2f",
        help="Annual volatility of the underlying"
    )

    st.divider()

    #Heatmap Configuration
    st.subheader("Heatmap Parameters")

    #Spot price range
    col5, col6 = st.columns(2)
    with col5:
        min_S = st.number_input(
            "Min Spot Price",
            min_value=0.01,
            value=S0 * 0.8,
            step=0.01,
            format="%.2f"
        )
    with col6:
        max_S = st.number_input(
            "Max Spot Price",
            min_value=0.01,
            value=S0 * 1.2,
            step=0.01,
            format="%.2f"
        )

    #Volatility range
    col7, col8 = st.columns(2)
    with col7:
        min_vol = st.number_input(
            "Min Volatility",
            min_value=0.01,
            max_value=2.0,
            value=max(0.01, sigma * 0.5),
            step=0.01,
            format="%.2f"
        )
    with col8:
        max_vol = st.number_input(
            "Max Volatility",
            min_value=0.01,
            max_value=2.0,
            value=min(2.0, sigma * 1.5),
            step=0.01,
            format="%.2f"
        )

    #Grid resolution
    col9, col10 = st.columns(2)
    with col9:
        nS = st.slider(
            "Spot Grid Points",
            min_value=10,
            max_value=100,
            value=50,
            step=5
        )
    with col10:
        nV = st.slider(
            "Vol Grid Points",
            min_value=10,
            max_value=100,
            value=50,
            step=5
        )

    st.divider()

    #Purchase Prices for P&L
    st.subheader("Purchase Prices (P&L)")

    col11, col12 = st.columns(2)
    with col11:
        purchase_call = st.number_input(
            "Call Purchase Price",
            min_value=0.0,
            value=10.0,
            step=0.01,
            format="%.2f")
    with col11:
        purchase_call = st.number_input(
            "Call Purchase Price",
            min_value=0.0,
            value=10.0,
            step=0.01,
            format="%.2f",
            help="Your purchase price for call option"
        )
    with col12:
        purchase_put = st.number_input(
            "Put Purchase Price",
            min_value=0.0,
            value=10.0,
            step=0.01,
            format="%.2f",
            help="Your purchase price for put option"
        )
    
    st.divider()
    
    #Database Configuration
    st.subheader("🗄️ Database Settings")
    
    db_enabled = st.checkbox("Enable Database Storage", value=True)
    
    if db_enabled:# ReTUn To THIS
        default_url = os.environ.get("MYSQL_URL", "mysql+pymysql://user:pass@localhost:3306/Evan_dev_db")
        mysql_url = st.text_input(
            "MySQL Connection URL",
            value=default_url,
            type="password",
            help="Format: mysql+pymysql://user:pass@host:port/database"
        )
        auto_create = st.checkbox("Auto-create tables", value=True)
    
    st.divider()
    
    #Additional Options
    st.subheader("⚡ Additional Options")
    show_greeks = st.checkbox("Calculate Greeks", value=True)
    show_values = st.checkbox("Show values on heatmap", value=False)
    show_3d = st.checkbox("Show 3D surface plots", value=False)

#Main content area
#Calculate button with custom styling
col_main1, col_main2, col_main3 = st.columns([1, 2, 1])
with col_main2:
    calculate = st.button("   Calculate Options", use_container_width=True, type="primary")

if calculate:
    #Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    #Step 1: Calculate base option prices and Greeks
    status_text.text("Calculating option prices...")
    progress_bar.progress(20)
    
    #Calculate with Greeks if requested
    if show_greeks:
        call_base, put_base, greeks = black_scholes_call_put(
            S0, K, r, sigma, T, compute_greeks=True
        )
    else:
        call_base, put_base = black_scholes_call_put(S0, K, r, sigma, T)
        greeks = None
    
    #Step 2: Display base results
    status_text.text("Displaying results...")
    progress_bar.progress(40)
    
    st.header("   Option Prices")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="call-option">', unsafe_allow_html=True)
        st.metric("Call Option Price", f"${float(call_base):.4f}")
        if greeks:
            st.caption(f"Delta: {float(greeks['call_delta']):.4f}")
            st.caption(f"Theta: {float(greeks['call_theta']):.4f}")
            st.caption(f"Rho: {float(greeks['call_rho']):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="put-option">', unsafe_allow_html=True)
        st.metric("Put Option Price", f"${float(put_base):.4f}")
        if greeks:
            st.caption(f"Delta: {float(greeks['put_delta']):.4f}")
            st.caption(f"Theta: {float(greeks['put_theta']):.4f}")
            st.caption(f"Rho: {float(greeks['put_rho']):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if greeks:
            st.metric("Gamma", f"{float(greeks['gamma']):.6f}")
            st.metric("Vega", f"{float(greeks['vega']):.4f}")
            st.caption("*Greeks shown for current parameters")
    
    #Step 3: Generate heatmap data
    status_text.text("Generating heatmaps...")
    progress_bar.progress(60)
    
    SS, VV, S_vec, V_vec = build_grids(min_S, max_S, min_vol, max_vol, nS, nV)
    call_surf, put_surf = black_scholes_call_put(SS, K, r, VV, T)
    
    #Step 4: Create visualizations
    status_text.text("Creating visualizations...")
    progress_bar.progress(80)
    
    #Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([" Price Heatmaps", " P&L Analysis", " Greeks Heatmaps", " 3D Surfaces"])
    
    with tab1:
        st.subheader("Option Price Heatmaps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            #Call price heatmap
            fig_call = px.imshow(
                call_surf,
                x=S_vec,
                y=V_vec,
                origin="lower",
                aspect="auto",
                labels=dict(x="Spot Price", y="Volatility", color="Call Price"),
                title="Call Option Prices",
                color_continuous_scale="RdYlGn",
                zmin=0,
                zmax=np.max(call_surf)
            )
            
            if show_values and nS <= 20 and nV <= 20:
                fig_call.update_traces(
                    text=np.round(call_surf, 2),
                    texttemplate="%{text}",
                    textfont_size=8
                )
            
            fig_call.update_layout(
                height=500,
                font=dict(size=12),
                hovermode='closest'
            )
            st.plotly_chart(fig_call, use_container_width=True)
        
        with col2:
            #Put price heatmap
            fig_put = px.imshow(
                put_surf,
                x=S_vec,
                y=V_vec,
                origin="lower",
                aspect="auto",
                labels=dict(x="Spot Price", y="Volatility", color="Put Price"),
                title="Put Option Prices",
                color_continuous_scale="RdYlGn",
                zmin=0,
                zmax=np.max(put_surf)
            )
            
            if show_values and nS <= 20 and nV <= 20:
                fig_put.update_traces(
                    text=np.round(put_surf, 2),
                    texttemplate="%{text}",
                    textfont_size=8
                )
            
            fig_put.update_layout(
                height=500,
                font=dict(size=12),
                hovermode='closest'
            )
            st.plotly_chart(fig_put, use_container_width=True)
    
    with tab2:
        st.subheader("Profit & Loss Analysis")
        
        #Calculate P&L surfaces
        pnl_call = pnl_surface(call_surf, purchase_call)
        pnl_put = pnl_surface(put_surf, purchase_put)
        
        col1, col2 = st.columns(2)
        
        with col1:
            #Call P&L heatmap
            fig_pnl_call = px.imshow(
                pnl_call,
                x=S_vec,
                y=V_vec,
                origin="lower",
                aspect="auto",
                labels=dict(x="Spot Price", y="Volatility", color="P&L"),
                title=f"Call P&L (Purchase: ${purchase_call:.2f})",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                zmin=np.min(pnl_call),
                zmax=np.max(pnl_call)
            )
            
            if show_values and nS <= 20 and nV <= 20:
                fig_pnl_call.update_traces(
                    text=np.round(pnl_call, 2),
                    texttemplate="%{text}",
                    textfont_size=8
                )
            
            fig_pnl_call.update_layout(
                height=500,
                font=dict(size=12),
                hovermode='closest'
            )
            st.plotly_chart(fig_pnl_call, use_container_width=True)
            
            #P&L statistics
            st.info(f"""
            **Call P&L Statistics:**
                Max Profit: ${np.max(pnl_call):.2f}
                Max Loss: ${np.min(pnl_call):.2f}
                Breakeven area: {np.sum(np.abs(pnl_call) < 0.01) / pnl_call.size * 100:.1f}%
            """)
        
        with col2:
            #Put P&L heatmap
            fig_pnl_put = px.imshow(
                pnl_put,
                x=S_vec,
                y=V_vec,
                origin="lower",
                aspect="auto",
                labels=dict(x="Spot Price", y="Volatility", color="P&L"),
                title=f"Put P&L (Purchase: ${purchase_put:.2f})",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                zmin=np.min(pnl_put),
                zmax=np.max(pnl_put)
            )
            
            if show_values and nS <= 20 and nV <= 20:
                fig_pnl_put.update_traces(
                    text=np.round(pnl_put, 2),
                    texttemplate="%{text}",
                    textfont_size=8
                )
            
            fig_pnl_put.update_layout(
                height=500,
                font=dict(size=12),
                hovermode='closest'
            )
            st.plotly_chart(fig_pnl_put, use_container_width=True)
            
            #P&L statistics
            st.info(f"""
            **Put P&L Statistics:**
                Max Profit: ${np.max(pnl_put):.2f}
                Max Loss: ${np.min(pnl_put):.2f}
                Breakeven area: {np.sum(np.abs(pnl_put) < 0.01) / pnl_put.size * 100:.1f}%
            """)
    
    with tab3:
        if show_greeks:
            st.subheader("Greeks Heatmaps")
            
            #Calculate Greeks for the grid
            _, _, greeks_grid = black_scholes_call_put(SS, K, r, VV, T, compute_greeks=True)
            
            #Create subplots for Greeks
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Call Delta', 'Put Delta', 'Gamma',
                              'Vega', 'Call Theta', 'Put Theta'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            #Add heatmaps for each Greek
            greeks_data = [
                ('call_delta', 1, 1, 'RdYlGn'),
                ('put_delta', 1, 2, 'RdYlGn_r'),
                ('gamma', 1, 3, 'Viridis'),
                ('vega', 2, 1, 'Plasma'),
                ('call_theta', 2, 2, 'RdYlGn_r'),
                ('put_theta', 2, 3, 'RdYlGn_r')
            ]
            
            for greek_name, row, col, colorscale in greeks_data:
                fig.add_trace(
                    go.Heatmap(
                        z=greeks_grid[greek_name],
                        x=S_vec,
                        y=V_vec,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(len=0.4, y=0.8 if row == 1 else 0.2, x=1.02 + (col-1)*0.05)
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="Spot Price", row=2)
            fig.update_yaxes(title_text="Volatility", col=1)
            
            fig.update_layout(
                height=800,
                title_text="Greeks Sensitivity Analysis",
                showlegend=False,
                font=dict(size=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enable 'Calculate Greeks' in the sidebar to view Greeks heatmaps.")
    
    with tab4:
        if show_3d:
            st.subheader("3D Surface Plots")
            
            col1, col2 = st.columns(2)
            
            with col1:
                #3D Call surface
                fig_3d_call = go.Figure(data=[go.Surface(
                    x=S_vec,
                    y=V_vec,
                    z=call_surf,
                    colorscale='RdYlGn',
                    name='Call Price'
                )])
                
                fig_3d_call.update_layout(
                    title="Call Option Price Surface",
                    scene=dict(
                        xaxis_title="Spot Price",
                        yaxis_title="Volatility",
                        zaxis_title="Option Price",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                    ),
                    height=500
                )
                st.plotly_chart(fig_3d_call, use_container_width=True)
            
            with col2:
                #3D Put surface
                fig_3d_put = go.Figure(data=[go.Surface(
                    x=S_vec,
                    y=V_vec,
                    z=put_surf,
                    colorscale='RdYlGn',
                    name='Put Price'
                )])
                
                fig_3d_put.update_layout(
                    title="Put Option Price Surface",
                    scene=dict(
                        xaxis_title="Spot Price",
                        yaxis_title="Volatility",
                        zaxis_title="Option Price",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                    ),
                    height=500
                )
                st.plotly_chart(fig_3d_put, use_container_width=True)
        else:
            st.info("Enable '3D surface plots' in the sidebar to view 3D visualizations.")
    
    #Step 5: Save to database
    if db_enabled:
        status_text.text("Saving to database...")
        progress_bar.progress(90)
        
        try:
            engine = create_engine_from_url(mysql_url)
            if auto_create:
                create_all(engine)
            
            #Prepare inputs dictionary
            inputs = {
                'StockPrice': S0,
                'StrikePrice': K,
                'InterestRate': r,
                'Volatility': sigma,
                'TimeToExpiry': T,
                'PurchaseCallPrice': purchase_call,
                'PurchasePutPrice': purchase_put,
                'MinSpot': min_S,
                'MaxSpot': max_S,
                'MinVol': min_vol,
                'MaxVol': max_vol,
                'GridNSpot': nS,
                'GridNVol': nV
            }
            
            #Insert calculation
            calc_id, n_rows = insert_calculation(
                engine, inputs, S_vec, V_vec, call_surf, put_surf
            )
            
            st.success(f"""
               **Data saved successfully!**
                Calculation ID: {calc_id}
                Total rows inserted: {n_rows}
                Timestamp: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)
            
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            st.info("Check your database connection settings and try again.")
    
    #Complete
    status_text.text("Complete!")
    progress_bar.progress(100)
    progress_bar.empty()
    status_text.empty()

#Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Black-Scholes Option Pricing Model | Built with Streamlit</p>
    <p style='font-size: 0.9em;'>   Real-time calculations |    Greeks analysis |    P&L visualization</p>
</div>
""", unsafe_allow_html=True)

#Information expander
with st.expander("Info About Black-Scholes Model"):
    st.markdown("""
    The **Black-Scholes model** is a mathematical framework for pricing Euro-style options.
    
    **Key Assumptions:**
        No arbitrage opportunities exist
        Markets are frictionless (no transaction costs or taxes)
        The risk-free rate is constant
        The underlying follows a geometric Brownian motion
        No dividends are paid during the option's life
    
    **The Greeks:**
        **Delta (Δ)**: Rate of change of option price with respect to underlying price
        **Gamma (Γ)**: Rate of change of delta with respect to underlying price
        **Theta (Θ)**: Rate of change of option price with respect to time
        **Vega (ν)**: Rate of change of option price with respect to volatility
        **Rho (ρ)**: Rate of change of option price with respect to interest rate
    
    **Database Schema:**
        Input parameters and calculation settings are stored in `BlackScholesInputs` table
        Grid results are stored in `BlackScholesOutputs` table with foreign key relationship
        All numerical values use DECIMAL(18,9) for precision
    """)

#Sidebar footer
with st.sidebar:
    st.divider()
    st.caption("Version 2.0 | Enhanced Edition")
    st.caption(f"Session started: {dt.now().strftime('%H:%M:%S')}")








