import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm

# ---------- Black–Scholes + Greeks ----------

def bs_price_greeks(S, K, T, r, sigma, option_type="call"):
    """Black–Scholes price and Greeks for European call/put (LONG option, per unit)."""
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
    if option_type == "call":
        intrinsic = max(S - K, 0.0)
    else:
        intrinsic = max(K - S, 0.0)
    time_value = max(model_price - intrinsic, 0.0)
    return intrinsic, time_value


def option_moneyness(S, K, option_type):
    eps = 1e-6
    if option_type == "call":
        if S > K + eps:
            return "ITM"
        elif abs(S - K) <= eps:
            return "ATM"
        else:
            return "OTM"
    else:
        if S < K - eps:
            return "ITM"
        elif abs(S - K) <= eps:
            return "ATM"
        else:
            return "OTM"


def long_payoff_array(S_grid, K, option_type, qty=1.0):
    if option_type == "call":
        return np.maximum(S_grid - K, 0.0) * qty
    else:
        return np.maximum(K - S_grid, 0.0) * qty


def pnl_from_long_payoff(long_payoff, premium_per_unit_now, qty=1.0, is_long=True):
    sign = 1 if is_long else -1
    return sign * long_payoff - sign * premium_per_unit_now * qty


# ---------- Streamlit app ----------

def main():
    st.set_page_config(page_title="Oil Option Payoff & Greeks", layout="wide")
    st.title("Oil Option Payoff, P&L and Greeks - by J Keites linkedin.com/in/jkeites")

    # Sidebar inputs
    st.sidebar.header("Vanilla European Option inputs")

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
    show_pnl_line = st.sidebar.checkbox(
        "Show P&L line (orange)", value=True, key="show_pnl_line"
    )
    show_pnl_shading = st.sidebar.checkbox(
        "Shade P&L area (red/green)", value=True, key="show_pnl_shading"
    )

    opt_type = "call" if option_side == "Call" else "put"
    is_long = position == "Long"
    pos_sign = 1 if is_long else -1
    T = T_days / 365.0

    # Zoom defaults
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

    # ---------- Pricing and Greeks (LONG option) ----------
    res_long = bs_price_greeks(S, K, T, r, sigma, opt_type)
    premium_per_unit_long = res_long["price"]

    # Position Greeks = long Greeks * +1 (long) or -1 (short)
    res_pos = {
        key: (val * pos_sign if key in ["delta", "gamma", "vega", "theta", "rho"] else val)
        for key, val in res_long.items()
    }

    signed_premium_per_unit_now = premium_per_unit_long * pos_sign
    premium_total_now = signed_premium_per_unit_now * qty

    intrinsic_now, time_val_now = intrinsic_and_time_value(
        S, K, opt_type, premium_per_unit_long
    )

    # ---------- 1) Position summary banner ----------
    moneyness = option_moneyness(S, K, opt_type)
    moneyness_long = (
        "In-the-money" if moneyness == "ITM"
        else "At-the-money" if moneyness == "ATM"
        else "Out-of-the-money"
    )

    st.markdown(
        f"""
        <div style="
            display:flex;
            flex-direction:row;
            align-items:center;
            justify-content:space-between;
            padding:10px 14px;
            margin-bottom:6px;
            border-radius:6px;
            background-color:#222222;
            color:#ffffff;
        ">
          <div style="font-size:22px; font-weight:bold;">
            {position} {K:g} {option_side}
          </div>
          <div style="font-size:16px;">
            Spot: <b>{S:g}</b> &nbsp;|&nbsp;
            Moneyness: <b>{moneyness}</b> ({moneyness_long})
          </div>
          <div style="font-size:14px;">
            T: <b>{T_days:g} days</b> &nbsp;|&nbsp;
            Vol: <b>{sigma*100:.2f}%</b> &nbsp;|&nbsp;
            r: <b>{r*100:.2f}%</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- 2) Premium block ----------
    st.markdown("### Premium details (per unit)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Position premium",
        f"{signed_premium_per_unit_now:.4f}",
        help="Positive = premium paid (long), negative = premium received (short).",
    )
    c2.metric("Intrinsic value (long)", f"{intrinsic_now:.4f}")
    c3.metric("Time value (long)", f"{time_val_now:.4f}")
    c4.metric("Total position premium", f"{premium_total_now:.4f}")

    st.caption(
        f"Strike {K:g}, underlying {S:g}, {T_days:g} days to expiry, "
        f"volatility {sigma*100:.2f}%."
    )

    # ---------- Payoff, P&L, premium vs underlying ----------
    st.subheader("Payoff and P&L at expiry")

    S_grid = np.linspace(base_min, base_max, 400)

    premium_grid_long = np.array(
        [bs_price_greeks(s_val, K, T, r, sigma, opt_type)["price"] for s_val in S_grid]
    )
    premium_grid_signed = premium_grid_long * pos_sign

    long_payoff = long_payoff_array(S_grid, K, opt_type, qty=qty)
    position_payoff = pos_sign * long_payoff
    pnl_expiry = pnl_from_long_payoff(
        long_payoff, premium_per_unit_long, qty=qty, is_long=is_long
    )

    df = pd.DataFrame(
        {
            "S_expiry": S_grid,
            "Position_payoff": position_payoff,
            "PnL": pnl_expiry,
            "Premium_signed": premium_grid_signed * qty,
        }
    )

    # Split P&L into positive and negative parts for shading
    df["PnL_pos"] = np.where(df["PnL"] > 0, df["PnL"], 0.0)
    df["PnL_neg"] = np.where(df["PnL"] < 0, df["PnL"], 0.0)

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

    # Shaded areas for P&L relative to 0
    pnl_area_pos = base.mark_area(
        color="green",
        opacity=0.25
    ).encode(
        y=alt.Y("PnL_pos:Q", title="P&L"),
        y2=alt.value(0)
    )

    pnl_area_neg = base.mark_area(
        color="red",
        opacity=0.25
    ).encode(
        y=alt.Y("PnL_neg:Q", title="P&L"),
        y2=alt.value(0)
    )

    # Start layers with payoff only; add shading and P&L conditionally
    layers = [payoff_line]

    if show_pnl_shading:
        layers.append(pnl_area_pos)
        layers.append(pnl_area_neg)

    if show_pnl_line:
        layers.append(pnl_line)

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

    # Strike label at top of graph
    strike_label_data = pd.DataFrame({
        "K": [K],
        "label": [f"Strike 'K' {K:g}"]
    })

    strike_label = (
        alt.Chart(strike_label_data)
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            color="red",
            fontSize=12,
            fontWeight="bold",
        )
        .encode(
            x="K:Q",
            y=alt.value(0),          # top of plotting area
            text="label:N",
        )
    )
    layers.append(strike_label)

    # Arrow showing current S and P&L at expiry
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

    # ---------- Greeks numeric table (POSITION Greeks) ----------
    st.subheader("Greeks (per unit for displayed position)")

    greeks_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Theta (per year)", "Rho"],
            "Value": [
                res_pos["delta"],
                res_pos["gamma"],
                res_pos["vega"],
                res_pos["theta"],
                res_pos["rho"],
            ],
        }
    )
    st.table(greeks_df)

    # ---------- Greeks explanation table ----------
    delta_val = res_pos["delta"]
    gamma_val = res_pos["gamma"]
    vega_val = res_pos["vega"]
    theta_val = res_pos["theta"]
    rho_val = res_pos["rho"]

    expl_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Explanation": [
                f"Delta = {delta_val:.4f}. Approximate change in P&L for a 1 $/bbl move in "
                f"the underlying, for this {position} {option_side} position.",
                f"Gamma = {gamma_val:.4f}. Rate of change of delta with respect to the underlying.",
                f"Vega = {vega_val:.4f}. Sensitivity of this position's value to volatility "
                f"(per 1.00 = 100 vol points). Per 1% vol point ≈ {vega_val/100.0:.4f}.",
                f"Theta = {theta_val:.4f} per year. Time decay for this position, "
                f"≈ {theta_val/365.0:.4f} per day.",
                f"Rho = {rho_val:.4f}. Sensitivity of this position's value to a 1%  change "
                f"in the risk-free rate; per 1 percentage point ≈ {rho_val/100.0:.4f}.",
            ],
        }
    )

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

    st.subheader("Greeks explanation (for this position)")
    st.markdown(
        expl_df.to_html(classes="greeks-expl-table", index=False),
        unsafe_allow_html=True,
    )


if 
