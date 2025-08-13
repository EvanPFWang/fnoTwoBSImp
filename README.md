# Black–Scholes App (REPL + Streamlit + MySQL)

## Quickstart

```bash
cd bs_black_scholes_app
python -m venv .venv && source .venv/bin/activate  # or use conda
pip install -r requirements.txt

# 1) CLI pricer
python repl_app.py

# 2) Streamlit UI
export MYSQL_URL="mysql+pymysql://user:pass@localhost:3306/Evan_dev_db"
streamlit run streamlit_app.py
```

## Tables

Use `schema.sql` to create the database and tables. Or allow the app to auto-create.

## Notes

- Prices are computed with the closed-form Black–Scholes formula (European options).
- The heatmap shows price as a function of (Spot, Volatility). Below it, P&L heatmaps for the call and the put use your purchase prices.
- On **Calculate**, the app writes one `BlackScholesInputs` row and N×M×2 `BlackScholesOutputs` rows (call + put for each grid cell).
- SQLAlchemy URL examples:
  - `mysql+pymysql://user:pass@localhost:3306/Evan_dev_db`
