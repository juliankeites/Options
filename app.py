import numpy as np
import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm

# --- Black–Scholes + Greeks ---
def bs_price_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0,
                "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho   = K * T * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho   = -K * T * exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega  = S * norm.pdf(d1) * sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T))
             - (r * K * exp(-r * T) * (norm.cdf(d2) if option_type=="call" else norm.cdf(-d2))))
    return {"price": price, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}

def intrinsic_and_time_value(S, K, option_type, model_price):
    if option_type == "call":
        intrinsic = max(S - K, 0.0)
    else:
        intrinsic = max(K - S, 0.0)
    time_value = max(model_price - intrinsic, 0.0)
    return intrinsic, time_value

def payoff_array(S_grid, K, option_type, qty=1):
    if option_type == "call":
        return np.maximum(S_grid - K, 0.0) * qty
    else:
        return np.maximum(K - S_grid, 0.0) * qty

def pnl_array(S_grid, payoff_expiry, premium, qty=1, is_long=True):
    sign = 1 if is_long else -1
    return payoff_expiry * sign - premium * qty * sign

# --- Streamlit UI ---
def main():
    st.title("Oil Option Payoff, P&L and Greeks")

    st.sidebar.header("Option inputs")
    S = st.sidebar.number_input("Current oil price (S)", value=80.0, min_value=0.01)
    K = st.sidebar.number_input("Strike (K)", value=80.0, min_value=0.01)
    T_days = st.sidebar.number_input("Days to expiry", value=30, min_value=1)
    sigma = st.sidebar.number_input("Volatility (annual, %)", value=30.0, min_value=1.0) / 100.0
    r = st.sidebar.number_input("Risk-free rate (annual, %)", value=0.0) / 100.0
    qty = st.sidebar.number_input("Quantity (bbl or lots)", value=1.0, min_value=0.01)
    option_side = st.sidebar.selectbox("Option type", ["Call", "Put"])
    position = st.sidebar.selectbox("Position", ["Long", "Short"])

    opt_type = "call" if option_side == "Call" else "put"
    is_long = (position == "Long")
    T = T_days / 365.0

    res = bs_price_greeks(S, K, T, r, sigma, opt_type)
    premium = res["price"] * qty
    intrinsic, time_val = intrinsic_and_time_value(S, K, opt_type, res["price"])

    st.subheader("Premium, intrinsic and time value")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model premium (per unit)", f"{res['price']:.4f}")
    col2.metric("Intrinsic value (per unit)", f"{intrinsic:.4f}")
    col3.metric("Time value (per unit)", f"{time_val:.4f}")

    st.write(f"Total premium for position ({position} {option_side}): **{premium:.4f}**")

    # Payoff & PnL at expiry
    S_grid = np.linspace(max(0.01, S * 0.2), S * 2.0, 100)
    payoff_expiry = payoff_array(S_grid, K, opt_type, qty=qty)
    pnl_expiry = pnl_array(S_grid, payoff_expiry, res["price"], qty=qty, is_long=is_long)

    st.subheader("Payoff and P&L at expiry")

    payoff_df = {
        "Underlying price at expiry": S_grid,
        "Gross payoff": payoff_expiry,
        "P&L (net of premium)": pnl_expiry,
    }
    st.line_chart(payoff_df, x="Underlying price at expiry")

    st.subheader("Greeks (per unit)")
    st.table({
        "Greek": ["Delta", "Gamma", "Vega", "Theta (per year)", "Rho"],
        "Call" if opt_type == "call" else "Put": [
            res["delta"],
            res["gamma"],
            res["vega"],
            res["theta"],
            res["rho"],
        ],
    })

if __name__ == "__main__":
    main()
