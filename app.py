import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm

# --- Black–Scholes + Greeks ---
def bs_price_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return {
            "price": 0.0,
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm.pdf(d1) * sqrt(T)
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * sqrt(T))
        - (
            r
            * K
            * exp(-r * T)
            * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))
        )
    )

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def intrinsic_and_time_value(S, K, option_type, model_price):
    if option_type == "call":
        intrinsic = max(S - K, 0.0)
    else:
        intrinsic = max(K - S, 0.0)
    time_value = max(model_price - intrinsic, 0.0)
    return intrinsic, time_value


def payoff_array(S_grid, K, option_type, qty=1.0):
    if option_type == "call":
        return np.maximum(S_grid - K, 0.0) * qty
    else:
        return np.maximum(K - S_grid, 0.0) * qty


def pnl_array(S_grid, payoff_expiry, premium_per_unit, qty=1.0, is_long=True):
    """
    P&L at expiry = +/- payoff -/+ premium, depending on long/short.
    Long:  PnL = payoff - premium
    Short: PnL = -payoff + premium
    """
    sign = 1 if is_long else -1
    return sign * payoff_expiry - sign * premium_per_unit * qty


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Oil Option Payoff & Greeks", layout="wide")
    st.title("Oil Option Payoff, P&L and Greeks")

    # Sidebar inputs
    st.sidebar.header("Option inputs")

    S = st.sidebar.number_input("Current oil price (S)", value=80.0, min_value=0.01)
    K = st.sidebar.number_input("Strike (K)", value=80.0, min_value=0.01)
    T_days = st.sidebar.number_input("Days to expiry", value=30, min_value=1)
    sigma = (
        st.sidebar.number_input("Volatility (annual, %)", value=30.0, min_value=1.0)
        / 100.0
    )
    r = st.sidebar.number_input("Risk-free rate (annual, %)", value=0.0) / 100.0
    qty = st.sidebar.number_input("Quantity (bbl or lots)", value=1.0, min_value=0.01)
    option_side = st.sidebar.selectbox("Option type", ["Call", "Put"])
    position = st.sidebar.selectbox("Position", ["Long", "Short"])

    opt_type = "call" if option_side == "Call" else "put"
    is_long = position == "Long"
    T = T_days / 365.0

    # Pricing + Greeks
    res = bs_price_greeks(S, K, T, r, sigma, opt_type)
    premium_per_unit = res["price"]
    premium_total = premium_per_unit * qty
    intrinsic, time_val = intrinsic_and_time_value(
        S, K, opt_type, premium_per_unit
    )

    # Premium, intrinsic, time value
    st.subheader("Premium, intrinsic and time value (per unit)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model premium", f"{premium_per_unit:.4f}")
    col2.metric("Intrinsic value", f"{intrinsic:.4f}")
    col3.metric("Time value", f"{time_val:.44f}")

    st.write(
        f"Total premium for the position ({position} {option_side}, qty {qty:g}): "
        f"**{premium_total:.4f}**"
    )

    # Payoff & PnL at expiry
    st.subheader("Payoff and P&L at expiry")

    # Price grid
    S_min = max(0.01, S * 0.2)
    S_max = S * 2.0
    S_grid = np.linspace(S_min, S_max, 200)

    # Compute payoff and P&L fresh each run
    payoff_expiry = payoff_array(S_grid, K, opt_type, qty=qty)
    pnl_expiry = pnl_array(
        S_grid, payoff_expiry, premium_per_unit, qty=qty, is_long=is_long
    )

    df = pd.DataFrame(
        {
            "S_expiry": S_grid,
            "Payoff": payoff_expiry,
            "PnL": pnl_expiry,
        }
    )

    base = alt.Chart(df).encode(
        x=alt.X("S_expiry", title="Underlying price at expiry")
    )

    payoff_line = base.mark_line(color="steelblue", strokeWidth=2).encode(
        y=alt.Y("Payoff", title="Value")
    )

    pnl_line = base.mark_line(color="orange", strokeWidth=2).encode(
        y="PnL"
    )

    # Vertical line at strike
    strike_line = (
        alt.Chart(pd.DataFrame({"K": [K]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(x="K:Q")
    )

    # Arrow marker at current underlying on P&L curve
    idx_closest = int(np.abs(S_grid - S).argmin())
    S_now = float(S_grid[idx_closest])
    pnl_now = float(pnl_expiry[idx_closest])

    arrow_point = (
        alt.Chart(pd.DataFrame({"S_now": [S_now], "PnL_now": [pnl_now]}))
        .mark_point(
            color="black",
            size=80,
            shape="triangle-up" if is_long else "triangle-down",
        )
        .encode(
            x="S_now:Q",
            y="PnL_now:Q",
            tooltip=[
                alt.Tooltip("S_now:Q", title="Underlying now"),
                alt.Tooltip("PnL_now:Q", title="PnL at expiry"),
            ],
        )
    )

    premium_label = (
        alt.Chart(pd.DataFrame({"x": [S_now], "y": [pnl_now]}))
        .mark_text(align="left", dx=5, dy=-10, color="black")
        .encode(
            x="x:Q",
            y="y:Q",
            text=alt.value(f"Premium now: {premium_per_unit:.4f}"),
        )
    )

    chart = (
        payoff_line
        + pnl_line
        + strike_line
        + arrow_point
        + premium_label
    ).properties(
        width=800,
        height=450,
        title="Payoff (blue) and P&L (orange) at expiry",
    )

    st.altair_chart(chart, use_container_width=True)

    # Greeks table
    st.subheader("Greeks (per unit)")
    greeks_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Theta (per year)", "Rho"],
            "Value": [
                res["delta"],
                res["gamma"],
                res["vega"],
                res["theta"],
                res["rho"],
            ],
        }
    )
    st.table(greeks_df)


if __name__ == "__main__":
    main()
