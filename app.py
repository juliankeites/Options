import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm

# ---------- Black–Scholes + Greeks ----------

def bs_price_greeks(S, K, T, r, sigma, option_type="call"):
    """Black–Scholes price and Greeks for European call/put (per unit)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
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
        cum2 = norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * exp(-r * T) * norm.cdf(-d2)
        cum2 = norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm.pdf(d1) * sqrt(T)
    theta = -S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * cum2

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def intrinsic_and_time_value(S, K, option_type, model_price):
    """Intrinsic and time value for a long option at current S."""
    if option_type == "call":
        intrinsic = max(S - K, 0.0)
    else:
        intrinsic = max(K - S, 0.0)
    time_value = max(model_price - intrinsic, 0.0)
    return intrinsic, time_value

def option_moneyness(S, K, option_type):
    """
    Return 'ITM', 'ATM', or 'OTM' for the current option.
    option_type: 'call' or 'put'
    """
    eps = 1e-6  # small tolerance
    if option_type == "call":
        if S > K + eps:
            return "ITM"
        elif abs(S - K) <= eps:
            return "ATM"
        else:
            return "OTM"
    else:  # put
        if S < K - eps:
            return "ITM"
        elif abs(S - K) <= eps:
            return "ATM"
        else:
            return "OTM"

def long_payoff_array(S_grid, K, option_type, qty=1.0):
    """Long option payoff at expiry for each S on grid (per specified quantity)."""
    if option_type == "call":
        return np.maximum(S_grid - K, 0.0) * qty
    else:
        return np.maximum(K - S_grid, 0.0) * qty


def pnl_from_long_payoff(long_payoff, premium_per_unit_now, qty=1.0, is_long=True):
    """
    P&L at expiry for chosen position:

    long:  PnL = long_payoff - premium_now * qty
    short: PnL = -long_payoff + premium_now * qty
    """
    sign = 1 if is_long else -1
    return sign * long_payoff - sign * premium_per_unit_now * qty


# ---------- Streamlit app ----------

def main():
    st.set_page_config(page_title="Oil Option Payoff & Greeks", layout="wide")
    st.title("Oil Option Payoff, P&L and Greeks")

    # Sidebar inputs
    st.sidebar.header("Option inputs")

    S = st.sidebar.number_input(
        "Current oil price (S)", value=80.0, min_value=0.01, key="S_input"
    )
    K = st.sidebar.number_input(
        "Strike (K)", value=80.0, min_value=0.01, key="K_input"
    )
    T_days = st.sidebar.number_input(
        "Days to expiry", value=30, min_value=1, key="T_days_input"
    )
    sigma = (
        st.sidebar.number_input(
            "Volatility (annual, %)", value=30.0, min_value=1.0, key="sigma_input"
        )
        / 100.0
    )
    r = (
        st.sidebar.number_input(
            "Risk-free rate (annual, %)", value=0.0, key="r_input"
        )
        / 100.0
    )
    qty = st.sidebar.number_input(
        "Quantity (bbl or lots)", value=1.0, min_value=0.01, key="qty_input"
    )
    option_side = st.sidebar.selectbox(
        "Option type", ["Call", "Put"], key="option_side"
    )
    position = st.sidebar.selectbox(
        "Position", ["Long", "Short"], key="position_side"
    )
    show_premium_line = st.sidebar.checkbox(
        "Show premium vs underlying", value=True, key="show_premium_line"
    )

    opt_type = "call" if option_side == "Call" else "put"
    is_long = position == "Long"
    T = T_days / 365.0

    # Default grid range for zoom sliders
    base_min = max(0.01, S * 0.2)
    base_max = S * 2.0

    st.sidebar.header("Zoom / axis range")
    S_min_zoom = st.sidebar.slider(
        "Min underlying on chart",
        min_value=float(base_min),
        max_value=float(base_max),
        value=float(base_min),
        key="zoom_min",
    )
    S_max_zoom = st.sidebar.slider(
        "Max underlying on chart",
        min_value=float(base_min),
        max_value=float(base_max),
        value=float(base_max),
        key="zoom_max",
    )
    if S_max_zoom <= S_min_zoom:
        S_max_zoom = S_min_zoom + 1e-6

    # Current price, intrinsic, time value, Greeks
    res_now = bs_price_greeks(S, K, T, r, sigma, opt_type)
    premium_per_unit_now = res_now["price"]          # always the long fair value per unit

  

    # Signed premium for this position (positive = cash outflow for long, negative = inflow for short)
    signed_premium_per_unit_now = premium_per_unit_now if is_long else -premium_per_unit_now

    premium_total_now = signed_premium_per_unit_now * qty
    intrinsic_now, time_val_now = intrinsic_and_time_value(
        S, K, opt_type, premium_per_unit_now
    )

    st.subheader("Premium, intrinsic and time value (per unit)")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Position premium (per unit)",
        f"{signed_premium_per_unit_now:.4f}",
        help="Positive = premium paid (long), negative = premium received (short).",
    )
    c2.metric("Intrinsic value (long)", f"{intrinsic_now:.4f}")
    c3.metric("Time value (long)", f"{time_val_now:.4f}")
    st.write(
        f"Total premium for the position ({position} {option_side}, qty {qty:g}): "
        f"**{premium_total_now:.4f}**"
    )

    # Payoff, P&L, premium vs underlying
    st.subheader("Payoff and P&L at expiry")

    # Grid over base range, chart zooms via sliders
    S_grid = np.linspace(base_min, base_max, 400)

    # Fair value (long premium) vs S on grid, then sign it for position
    premium_grid_long = np.array(
        [bs_price_greeks(s_val, K, T, r, sigma, opt_type)["price"] for s_val in S_grid]
    )
    premium_grid_signed = np.where(is_long, premium_grid_long, -premium_grid_long)

    # Long payoff, then convert to position payoff and P&L
    long_payoff = long_payoff_array(S_grid, K, opt_type, qty=qty)
    position_sign = 1 if is_long else -1
    position_payoff = position_sign * long_payoff
    pnl_expiry = pnl_from_long_payoff(
        long_payoff, premium_per_unit_now, qty=qty, is_long=is_long
    )

    df = pd.DataFrame(
        {
            "S_expiry": S_grid,
            "Position_payoff": position_payoff,        # long or short payoff
            "PnL": pnl_expiry,                         # includes premium income/cost
            "Premium_signed": premium_grid_signed * qty,  # position premium vs S (negative for shorts)
        }
    )

    df_zoom = df[(df["S_expiry"] >= S_min_zoom) & (df["S_expiry"] <= S_max_zoom)]

    base = alt.Chart(df_zoom).encode(
        x=alt.X(
            "S_expiry",
            title="Underlying price at expiry",
            scale=alt.Scale(domain=[S_min_zoom, S_max_zoom]),
        )
    )

    payoff_line = base.mark_line(color="steelblue", strokeWidth=2).encode(
        y=alt.Y("Position_payoff", title="Value"),
        tooltip=[
            alt.Tooltip("S_expiry:Q", title="S at expiry"),
            alt.Tooltip("Position_payoff:Q", title=f"{position} {option_side} payoff"),
        ],
    )

    pnl_line = base.mark_line(color="orange", strokeWidth=2).encode(
        y=alt.Y("PnL", title="P&L"),
        tooltip=[
            alt.Tooltip("S_expiry:Q", title="S at expiry"),
            alt.Tooltip("PnL:Q", title=f"{position} {option_side} P&L"),
        ],
    )

    layers = [payoff_line, pnl_line]

    if show_premium_line:
        premium_line = base.mark_line(color="green", strokeDash=[4, 2]).encode(
            y=alt.Y("Premium_signed", title="Position premium vs S"),
            tooltip=[
                alt.Tooltip("S_expiry:Q", title="S"),
                alt.Tooltip("Premium_signed:Q", title="Position premium vs S"),
            ],
        )
        layers.append(premium_line)

    # Strike line
    strike_line = (
        alt.Chart(pd.DataFrame({"K": [K]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(x=alt.X("K:Q", scale=alt.Scale(domain=[S_min_zoom, S_max_zoom])))
    )
    layers.append(strike_line)

    # Arrow at current S on P&L curve, if inside zoom window
    idx_closest = int(np.abs(S_grid - S).argmin())
    S_now = float(S_grid[idx_closest])
    pnl_now = float(pnl_expiry[idx_closest])

    if S_min_zoom <= S_now <= S_max_zoom:
        arrow_point = (
            alt.Chart(pd.DataFrame({"S_now": [S_now], "PnL_now": [pnl_now]}))
            .mark_point(
                color="white",
                size=80,
                shape="triangle-up" if is_long else "triangle-down",
            )
            .encode(
                x="S_now:Q",
                y="PnL_now:Q",
                tooltip=[
                    alt.Tooltip("S_now:Q", title="Underlying now"),
                    alt.Tooltip("PnL_now:Q", title="P&L at expiry"),
                ],
            )
        )
        layers.append(arrow_point)

        label_side = "long" if is_long else "short"
        premium_label = (
            alt.Chart(pd.DataFrame({"x": [S_now], "y": [pnl_now]}))
            .mark_text(align="left", dx=50, dy=-10, color="White")
            .encode(
                x="x:Q",
                y="y:Q",
                text=alt.value(
                    f"Premium now ({label_side}): {signed_premium_per_unit_now:.4f}"
                ),
            )
        )
        layers.append(premium_label)

    chart = alt.layer(*layers).properties(
        width=800,
        height=450,
        title="Payoff (blue), P&L (orange), Position premium (green) vs underlying",
    )

    st.altair_chart(chart, use_container_width=True)

    # ---------- Greeks numeric table ----------
    st.subheader("Greeks (per unit at current underlying, long option)")

    greeks_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Theta (per year)", "Rho"],
            "Value": [
                res_now["delta"],
                res_now["gamma"],
                res_now["vega"],
                res_now["theta"],
                res_now["rho"],
            ],
        }
    )
    st.table(greeks_df)

    # ---------- Greeks explanation table ----------
    # Build text with current values interpolated
    delta_val = res_now["delta"]
    gamma_val = res_now["gamma"]
    vega_val = res_now["vega"]
    theta_val = res_now["theta"]
    rho_val = res_now["rho"]

    expl_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Explanation": [
                f"Delta = {delta_val:.4f}. Approximate change in option value "
                f"for a 1-unit move in the underlying price. For example, if the underlying moves up by 1, "
                f"the option value changes by about {delta_val:.4f}.",
                f"Gamma = {gamma_val:.4f}. Rate of change of delta with respect to the underlying. "
                f"A 1-unit move in the underlying changes delta by about {gamma_val:.4f}.",
                f"Vega = {vega_val:.4f}. Sensitivity of option value to volatility. "
                f"A 1 percentage-point increase in implied volatility changes the option value by roughly "
                f"{vega_val/100.0:.4f} (per 1 vol point).",
                f"Theta = {theta_val:.4f} per year. Sensitivity of option value to the passage of time, "
                f"holding other inputs constant. Per day (divide Theta by 365) this is about {theta_val/365.0:.4f}.",
                f"Rho = {rho_val:.4f}. Sensitivity of option value to the risk-free interest rate, and is 0% is used on Options on Forwards. "
                f"A 1 percentage-point increase in the rate changes the option value by roughly "
                f"{rho_val/100.0:.4f}.",
            ],
        }
    )

    # CSS for wrapping explanation text
    st.markdown(
        """
        <style>
        .greeks-expl-table {
            font-size: 11px;
            border-collapse: collapse;
            width: 100%;
        }
        .greeks-expl-table th, .greeks-expl-table td {
            border: 1px solid #ddd;
            padding: 4px 6px;
            white-space: normal;
            word-wrap: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Greeks explanation")
    st.markdown(
        expl_df.to_html(classes="greeks-expl-table", index=False),
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
