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
    # pliki JSON z depozytami / wypłatami
    df = pd.read_json(path)
    return df


def prepare_common_cols(df: pd.DataFrame) -> pd.DataFrame:
    # upewniamy się, że mamy potrzebne kolumny
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
    # liczba statusów per kraj / metoda
    table = (
        df.groupby([group_col, "status"])
          .size()
          .unstack(fill_value=0)
    )

    # bezpieczeństwo – jak nie ma accepted / rejected, dodajemy kolumny 0
    for col in ["accepted", "rejected"]:
        if col not in table.columns:
            table[col] = 0

    table["total"] = table.sum(axis=1)

    denom = (table["accepted"] + table["rejected"]).replace(0, pd.NA)
    table["success_rate_final"] = table["accepted"] / denom

    return table.reset_index()


def compute_durations(df: pd.DataFrame):
    """
    Liczymy czas od is_verify_code -> accepted
    dla (user_id, payment_name).
    createdAt zakładamy w milisekundach.
    """

    if "createdAt" not in df.columns or "status" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df_sorted = df.sort_values(["user_id", "payment_name", "createdAt"])
    durations = []
    last_verify = {}

    for row in df_sorted.itertuples():
        key = (row.user_id, row.payment_name)
        if row.status == "is_verify_code":
            # zapamiętujemy czas wysłania kodu
            last_verify[key] = row.createdAt
        elif row.status == "accepted":
            # gdy mamy accepted i wcześniejszy verify_code -> liczymy różnicę
            if key in last_verify:
                dt_ms = row.createdAt - last_verify[key]
                # filtr na dziwne outliery (> 1 dzień)
                if 0 < dt_ms < 24 * 3600 * 1000:
                    durations.append(
                        (key[0], key[1], row.createdAt, dt_ms)
                    )
                # czyścimy
                del last_verify[key]

    if not durations:
        return pd.DataFrame(), pd.DataFrame()

    dur_df = pd.DataFrame(
        durations,
        columns=["user_id", "payment_name", "accepted_createdAt", "duration_ms"]
    )
    dur_df["duration_sec"] = dur_df["duration_ms"] / 1000.0

    # podłączamy kraj z tabeli accepted
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
# UI – JEDEN BLOK DASHBOARDU (Deposit / Withdrawal)
# ---------------------------------------------------------

def render_section(df: pd.DataFrame, label: str):
    st.header(f"{label} – Transaction Dashboard")

    # -------- RAW DATA PREVIEW --------
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(200))

    # -------- OVERALL STATUS --------
    st.subheader("Overall Transaction Status")

    status_overall = compute_status_overall(df)

    col1, col2 = st.columns([1, 2])

    with col1:
        total_tx = int(status_overall["count"].sum())
        st.metric("Total transactions", f"{total_tx:,}")

        success_row = status_overall[status_overall["status"] == "accepted"]
        if not success_row.empty:
            success_rate = float(success_row["share"].iloc[0]) * 100
            st.metric("Success rate (overall)", f"{success_rate:.2f}%")

    with col2:
        st.bar_chart(
            status_overall.set_index("status")["count"],
            use_container_width=True,
        )

    # -------- SUCCESS RATE BY COUNTRY --------
    st.subheader("Success Rate by Country")

    s_country = compute_status_by_group(df, "country_name").sort_values(
        "success_rate_final"
    )
    st.dataframe(s_country)

    # wykres – słupki poziome
    st.bar_chart(
        s_country.set_index("country_name")["success_rate_final"],
        use_container_width=True,
    )

    # -------- SUCCESS RATE BY PAYMENT METHOD --------
    st.subheader("Success Rate by Payment Method")

    s_payment = compute_status_by_group(df, "payment_name").sort_values(
        "success_rate_final"
    )
    st.dataframe(s_payment)

    st.bar_chart(
        s_payment.set_index("payment_name")["success_rate_final"],
        use_container_width=True,
    )

    # -------- DURATIONS --------
    st.subheader("Transaction Duration (is_verify_code → accepted)")

    by_country, by_payment = compute_durations(df)

    if by_country.empty and by_payment.empty:
        st.info("No valid duration data could be computed for this dataset.")
        return

    # Duration by country
    if not by_country.empty:
        st.markdown("**Median Duration by Country (seconds)**")
        by_country_sorted = by_country.sort_values("median", ascending=False)
        st.dataframe(by_country_sorted)
        st.bar_chart(
            by_country_sorted.set_index("country_name")["median"],
            use_container_width=True,
        )

    # Duration by payment method
    if not by_payment.empty:
        st.markdown("**Median Duration by Payment Method (seconds)**")
        by_payment_sorted = by_payment.sort_values("median", ascending=False)
        st.dataframe(by_payment_sorted)
        st.bar_chart(
            by_payment_sorted.set_index("payment_name")["median"],
            use_container_width=True,
        )


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------

def main():
    st.title("📊 Transaction Dashboard")

    view = st.sidebar.selectbox(
        "Select Report Section:",
        ["Deposit", "Withdrawal"],
    )

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
