import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

st.set_page_config(page_title="Transaction Dashboard", layout="wide")

DEPOSIT_PATH = "deposit.json"
WITHDRAW_PATH = "withdraw_2024_nov2025_clean.json"

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path)
    return df


def prepare_common_cols(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "id",
        "user_id",
        "createdAt",
        "status",
        "payment_id",
        "country_id",
        "country_name",
        "payment_name",
        "method",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in data: {missing}")
    return df


# ---------------------------------------------------------
# AGGREGATIONS
# ---------------------------------------------------------

def compute_status_overall(df: pd.DataFrame) -> pd.DataFrame:
    out = df["status"].value_counts().reset_index()
    out.columns = ["status", "count"]
    out["share"] = out["count"] / out["count"].sum()
    return out


def compute_status_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    table = (
        df.groupby([group_col, "status"])
          .size()
          .unstack(fill_value=0)
    )

    # Ensure key status columns exist
    for col in ["accepted", "rejected", "pending", "processing", "is_verify_code"]:
        if col not in table.columns:
            table[col] = 0

    table["total"] = table.sum(axis=1)

    # Combined success: accepted + pending / total
    table["success_rate_combined"] = (table["accepted"] + table["pending"]) / table["total"].replace(0, pd.NA)

    # Strict success: accepted / (accepted + rejected)
    denom_strict = (table["accepted"] + table["rejected"]).replace(0, pd.NA)
    table["success_rate_strict"] = table["accepted"] / denom_strict

    return table.reset_index()


def compute_durations(df: pd.DataFrame):
    """
    Duration from is_verify_code -> accepted for (user_id, payment_name).
    createdAt assumed in milliseconds.
    """
    if "createdAt" not in df.columns or "status" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df_sorted = df.sort_values(["user_id", "payment_name", "createdAt"])

    durations = []
    last_verify = {}

    for row in df_sorted.itertuples():
        key = (row.user_id, row.payment_name)

        if row.status == "is_verify_code":
            last_verify[key] = row.createdAt

        elif row.status == "accepted":
            if key in last_verify:
                dt_ms = row.createdAt - last_verify[key]
                if 0 < dt_ms < 24 * 3600 * 1000:  # filter outliers > 1 day
                    durations.append((row.user_id, row.payment_name, row.createdAt, dt_ms))
                del last_verify[key]

    if not durations:
        return pd.DataFrame(), pd.DataFrame()

    dur_df = pd.DataFrame(
        durations,
        columns=["user_id", "payment_name", "accepted_createdAt", "duration_ms"]
    )
    dur_df["duration_sec"] = dur_df["duration_ms"] / 1000.0

    accepted = df[df["status"] == "accepted"][
        ["user_id", "payment_name", "createdAt", "country_name"]
    ]

    merged = dur_df.merge(
        accepted,
        left_on=["user_id", "payment_name", "accepted_createdAt"],
        right_on=["user_id", "payment_name", "createdAt"],
        how="left"
    )

    by_payment = (
        merged.groupby("payment_name")["duration_sec"]
        .agg(["count", "median", "mean", "max"])
        .reset_index()
    )

    by_country = (
        merged.groupby("country_name")["duration_sec"]
        .agg(["count", "median", "mean", "max"])
        .reset_index()
    )

    return by_country, by_payment


# ---------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------

def safe_int_from_status(status_overall: pd.DataFrame, status_name: str) -> int:
    if status_overall.empty:
        return 0
    return int(status_overall.loc[status_overall["status"] == status_name, "count"].sum())


def render_kpi_cards(status_overall: pd.DataFrame):
    total_tx = int(status_overall["count"].sum()) if not status_overall.empty else 0
    accepted_count = safe_int_from_status(status_overall, "accepted")
    pending_count = safe_int_from_status(status_overall, "pending")
    rejected_count = safe_int_from_status(status_overall, "rejected")

    combined_rate = ((accepted_count + pending_count) / total_tx * 100) if total_tx else 0.0
    strict_denom = accepted_count + rejected_count
    strict_rate = ((accepted_count / strict_denom) * 100) if strict_denom else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", f"{total_tx:,}")
    c2.metric("Accepted", f"{accepted_count:,}")
    c3.metric("Pending", f"{pending_count:,}")
    c4.metric("Rejected", f"{rejected_count:,}")
    c5.metric("Success (A+P)", f"{combined_rate:.2f}%")

    st.caption(f"Strict success = Accepted / (Accepted + Rejected): **{strict_rate:.2f}%**")


# ---------------------------------------------------------
# UI – ONE SECTION (Deposit / Withdrawal)
# ---------------------------------------------------------

def render_section(df: pd.DataFrame, label: str):
    st.header(f"{label} – Transaction Dashboard")

    # Sidebar controls (affect charts)
    st.sidebar.markdown("### Display options")
    rate_mode = st.sidebar.radio(
        "Success rate metric",
        ["Combined (Accepted + Pending) / Total", "Strict Accepted / (Accepted + Rejected)"],
        index=0
    )
    rate_col = "success_rate_combined" if rate_mode.startswith("Combined") else "success_rate_strict"

    min_n = st.sidebar.number_input("Min sample size (count)", min_value=1, value=20, step=1)
    top_n = st.sidebar.slider("Top N items in charts", min_value=5, max_value=30, value=15, step=1)

    # -------- RAW DATA PREVIEW --------
    with st.expander("Raw Data Preview (first 200 rows)"):
        st.dataframe(df.head(200))

    # -------- OVERALL STATUS --------
    st.subheader("Overall Transaction Status")
    status_overall = compute_status_overall(df)
    render_kpi_cards(status_overall)

    st.bar_chart(
        status_overall.set_index("status")["count"],
        use_container_width=True,
    )

    # -------- SUCCESS RATE BY COUNTRY --------
    st.subheader("Success Rate by Country")

    s_country = compute_status_by_group(df, "country_name")
    # filter by sample size
    s_country = s_country[s_country["total"] >= min_n]
    s_country = s_country.sort_values(rate_col).head(top_n)

    st.dataframe(
        s_country[["country_name", "total", "accepted", "pending", "rejected", rate_col]].rename(
            columns={rate_col: "success_rate"}
        )
    )

    st.bar_chart(
        s_country.set_index("country_name")[rate_col],
        use_container_width=True,
    )

    # -------- SUCCESS RATE BY PAYMENT METHOD --------
    st.subheader("Success Rate by Payment Method")

    s_payment = compute_status_by_group(df, "payment_name")
    s_payment = s_payment[s_payment["total"] >= min_n]
    s_payment = s_payment.sort_values(rate_col).head(top_n)

    st.dataframe(
        s_payment[["payment_name", "total", "accepted", "pending", "rejected", rate_col]].rename(
            columns={rate_col: "success_rate"}
        )
    )

    st.bar_chart(
        s_payment.set_index("payment_name")[rate_col],
        use_container_width=True,
    )

    # -------- DURATIONS --------
    st.subheader("Transaction Duration (is_verify_code → accepted)")

    by_country, by_payment = compute_durations(df)

    if by_country.empty and by_payment.empty:
        st.info("No valid duration data could be computed for this dataset.")
        return

    if not by_country.empty:
        st.markdown("**Median Duration by Country (seconds)**")
        bc = by_country[by_country["count"] >= min_n].sort_values("median", ascending=False).head(top_n)
        st.dataframe(bc)
        st.bar_chart(bc.set_index("country_name")["median"], use_container_width=True)

    if not by_payment.empty:
        st.markdown("**Median Duration by Payment Method (seconds)**")
        bp = by_payment[by_payment["count"] >= min_n].sort_values("median", ascending=False).head(top_n)
        st.dataframe(bp)
        st.bar_chart(bp.set_index("payment_name")["median"], use_container_width=True)


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------

def main():
    st.title("📊 Transaction Dashboard")

    view = st.sidebar.selectbox("Select Report Section:", ["Deposit", "Withdrawal"])

    if view == "Deposit":
        df = load_data(DEPOSIT_PATH)
        df = prepare_common_cols(df)
        render_section(df, "💰 Deposit")
    else:
        df = load_data(WITHDRAW_PATH)
        df = prepare_common_cols(df)
        render_section(df, "💸 Withdrawal")


if __name__ == "__main__":
    main()
