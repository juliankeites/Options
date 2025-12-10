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
    st.title("Oil Option Payoff, P&L and Greeks")

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
            {position} {option_side} @ {K:g}
          </div>
          <div style="font-size:16px;">
            Spot: <b>{S:g}</b> &nbsp;|&nbsp;
            Moneyness: <b>{moneyness}</b> ({moneyness_long})
          </div>
          <div style="font-size:14px;">
            T: <b>{T_days:g} days</b> &nbsp;|&nbsp;
            Vol: <b>{sigma*100:.2f}%</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- 2) Premium / intrinsic / time value row ----------
    st.markdown("### Premium details (per unit)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Position premium",
        f"{signed_premium_per_unit_now:.4f}",
        help="Positive = premium paid (long), negative = premium received (short).",
    )
    col2.metric("Intrinsic value (long)", f"{intrinsic_now:.4f}")
    col3.metric("Time value (long)", f"{time_val_now:.4f}")
    col4.metric("Total position premium", f"{premium_total_now:.4f}")

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

    strike_line = (
        alt.Chart(pd.DataFrame({"K": [K]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(x=alt.X("K:Q", scale=alt.Scale(domain=[S_min_zoom, S_max_zoom])))
    )
    layers.append(strike_line)

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
                f"Delta = {delta_val:.4f}. Approximate change in P&L for a 1-unit move in the underlying, "
                f"for this {position} {option_side} position.",
                f"Gamma = {gamma_val:.4f}. Rate of change of delta with respect to the underlying.",
                f"Vega = {vega_val:.4f}. Sensitivity of this position's value to volatility "
                f"(per 1.00 = 100 vol points). Per 1 vol point ≈ {vega_val/100.0:.4f}.",
                f"Theta = {theta_val:.4f} per year. Time decay for this position, "
                f"≈ {theta_val/365.0:.4f} per day.",
                f"Rho = {rho_val:.4f}. Sensitivity of this position's value to a 1.00 change "
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


if __name__ == "__main__":
    main()
