"""Real Estate Investment Advisor - Streamlit app (rule-based, no machine learning).

Run:  streamlit run app.py
Data: put india_housing_prices.csv next to this file (or upload it in the sidebar).
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import logic as L

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(HERE, "india_housing_prices.csv"),
              os.path.join(HERE, "data", "india_housing_prices.csv"),
              "india_housing_prices.csv", "data/india_housing_prices.csv"]


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading and cleaning data…")
def get_data(source):
    buf = source if isinstance(source, str) else io.BytesIO(source)
    return L.prepare(pd.read_csv(buf))


@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def load():
    path = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if path:
        return get_data(path)
    st.sidebar.warning("india_housing_prices.csv not found next to app.py.")
    up = st.sidebar.file_uploader("Upload india_housing_prices.csv", type="csv")
    if up is None:
        st.title("🏠 Real Estate Investment Advisor")
        st.info("Place **india_housing_prices.csv** in the app folder or upload it from the sidebar to begin.")
        st.stop()
    return get_data(up.getvalue())


df, ref, quality = load()
fmt_inr = lambda v: f"₹ {v:,.1f} L"

PAGES = ["📊 Overview", "🧹 Data & Preprocessing", "🔎 EDA (20 questions)",
         "🧮 Investment Advisor", "🗂️ Property Explorer", "ℹ️ Method & Limitations"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.caption(f"{len(df):,} cleaned property records")


def bar_note(text):
    st.caption(f"💡 {text}")


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("🏠 Real Estate Investment Advisor")
    st.write("Predicting property **profitability** and **future value** with transparent business rules "
             "(no machine learning).")
    c = st.columns(5)
    c[0].metric("Properties", f"{len(df):,}")
    c[1].metric("Cities / States", f"{df['City'].nunique()} / {df['State'].nunique()}")
    c[2].metric("Avg price", fmt_inr(df["Price_in_Lakhs"].mean()))
    c[3].metric("Avg price / sq ft", f"₹ {df['Price_per_SqFt_calc'].mean():,.0f}")
    c[4].metric("Rated 'Good Investment'", f"{df['Good_Investment'].mean():.1%}")

    a, b = st.columns(2)
    with a:
        g = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(g, x="Price_in_Lakhs", y="City", orientation="h", title="Top 10 cities by average price (₹ L)")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=380)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        g = df.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(g, x="Good_Investment", y="City", orientation="h",
                     title="Top 10 cities by share of Good Investments")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=380, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    heat = df.pivot_table(index="State", columns="Property_Type", values="Price_per_SqFt_calc", aggfunc="mean")
    fig = px.imshow(heat, text_auto=".0f", aspect="auto", color_continuous_scale="YlGnBu",
                    title="Location heatmap · average ₹ per sq ft (state × property type)")
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Use **Investment Advisor** to score a property, **Property Explorer** to filter listings, "
            "and **EDA** for the 20 analysis questions.")


def page_data():
    st.title("🧹 Data & Preprocessing")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{quality['rows_raw']:,}")
    c[1].metric("Missing values", quality["missing_total"])
    c[2].metric("Duplicate rows", quality["duplicate_rows"])
    c[3].metric("Rows after cleaning", f"{quality['rows_clean']:,}")

    st.subheader("Data quality findings")
    st.table(pd.DataFrame({
        "Check": ["Floor_No > Total_Floors (impossible)", "Price_per_SqFt = 0 after rounding",
                  "Price/sq ft IQR outliers (flagged, not removed)"],
        "Result": [f"{quality['floor_gt_total']:.1%} of rows → flagged, Floor_No capped at Total_Floors",
                   f"{quality['ppsf_zero_after_rounding']:.1%} of rows → recomputed in ₹ per sq ft",
                   f"{quality['ppsf_outliers']:,} rows ({quality['ppsf_outliers'] / quality['rows_clean']:.1%})"],
    }))

    st.subheader("Preprocessing steps")
    st.markdown("""
1. **Missing values / duplicates** – median (numeric) / mode (categorical) imputation and duplicate removal (no-ops on this dataset, kept for robustness).
2. **Floors** – `Floor_Data_Inconsistent` flag; `Floor_No` capped at `Total_Floors`.
3. **Price per sq ft** – recomputed in ₹/sq ft from price and size (the original is rounded to 2 decimals of lakhs).
4. **Encoding** – `Parking_Space` and `Security` → 0/1; amenities split into counts and `Has_Gym`, `Has_Pool`, … flags.
5. **Scaling** – school/hospital counts min–max scaled to 0–1 density scores; transport mapped Low/Medium/High → ⅓/⅔/1.
6. **Feature engineering** – `Infra_Score`, `Floor_Ratio`, `Size_per_BHK`, `City_Median_PPSF`, `Price_vs_City_Median`.
7. **Targets (rule-based)** – `Est_Annual_Growth_Pct`, `Future_Price_5Y`, `Good_Investment` (see *Method*).
""")
    st.subheader("Cleaned data preview")
    st.dataframe(df.head(200), use_container_width=True, height=300)
    st.download_button("⬇️ Download cleaned dataset (CSV)", to_csv_bytes(df),
                       file_name="india_housing_cleaned.csv", mime="text/csv")


def page_eda():
    st.title("🔎 Exploratory Data Analysis")
    sample = df.sample(min(len(df), 6000), random_state=42)
    t1, t2, t3, t4, t5 = st.tabs(["1–5 Price & Size", "6–10 Location", "11–15 Relationships",
                                  "16–20 Investment & Amenities", "Extras"])

    # ---------------- 1-5
    with t1:
        st.subheader("Q1 · Distribution of property prices")
        st.plotly_chart(px.histogram(df, x="Price_in_Lakhs", nbins=50, marginal="box",
                                     labels={"Price_in_Lakhs": "Price (₹ lakhs)"}), use_container_width=True)
        st.subheader("Q2 · Distribution of property sizes")
        st.plotly_chart(px.histogram(df, x="Size_in_SqFt", nbins=50, marginal="box",
                                     color_discrete_sequence=["#1f9d6b"]), use_container_width=True)
        st.subheader("Q3 · Price per sq ft by property type")
        a, b = st.columns(2)
        a.plotly_chart(px.box(sample, x="Property_Type", y="Price_per_SqFt_calc", points=False,
                              labels={"Price_per_SqFt_calc": "₹ per sq ft"}), use_container_width=True)
        a2 = df.groupby("Property_Type")["Price_per_SqFt_calc"].agg(["mean", "median"]).round(0)
        b.dataframe(a2, use_container_width=True)
        st.subheader("Q4 · Size vs price")
        a, b = st.columns(2)
        a.plotly_chart(px.scatter(sample, x="Size_in_SqFt", y="Price_in_Lakhs", opacity=0.35,
                                  title="Size vs price (6,000-row sample)"), use_container_width=True)
        b.plotly_chart(px.scatter(sample, x="Size_in_SqFt", y="Price_per_SqFt_calc", opacity=0.35, log_y=True,
                                  title="Size vs ₹ per sq ft (log scale)"), use_container_width=True)
        c = st.columns(2)
        c[0].metric("Correlation size ↔ price", f"{df['Size_in_SqFt'].corr(df['Price_in_Lakhs']):.3f}")
        c[1].metric("Correlation size ↔ price/sq ft", f"{df['Size_in_SqFt'].corr(df['Price_per_SqFt_calc']):.3f}")
        bar_note("Size and total price are almost uncorrelated; price per sq ft falls with size mostly because of the division.")
        st.subheader("Q5 · Outliers")
        rows = []
        for col in ["Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_calc"]:
            lo, hi = L.iqr_bounds(df[col])
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            rows.append({"feature": col, "lower bound": round(lo, 1), "upper bound": round(hi, 1),
                         "outliers": n, "share": f"{n / len(df):.1%}"})
        st.table(pd.DataFrame(rows).set_index("feature"))
        st.plotly_chart(px.box(sample, y="Price_per_SqFt_calc", title="₹ per sq ft (outliers shown)"),
                        use_container_width=True)

    # ---------------- 6-10
    with t2:
        st.subheader("Q6 · Average price per sq ft by state")
        s = df.groupby("State")["Price_per_SqFt_calc"].mean().sort_values().reset_index()
        fig = px.bar(s, x="Price_per_SqFt_calc", y="State", orientation="h")
        fig.update_layout(height=550); st.plotly_chart(fig, use_container_width=True)
        st.subheader("Q7 · Average property price by city")
        n = st.slider("Cities to show", 5, df["City"].nunique(), 20)
        c = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(n).reset_index()
        fig = px.bar(c, x="Price_in_Lakhs", y="City", orientation="h")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=max(400, 22 * n))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Q8 · Median age of properties by locality")
        la = df.groupby("Locality")["Age_of_Property"].median().sort_values()
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(la, nbins=20, labels={"value": "Median age (years)"}), use_container_width=True)
        ext = pd.concat([la.head(5), la.tail(5)]).reset_index()
        ext.columns = ["Locality", "Median age"]
        b.dataframe(ext, use_container_width=True)
        st.subheader("Q9 · BHK distribution across cities")
        bhk = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
        fig = px.imshow(bhk, text_auto=".0f", aspect="auto", color_continuous_scale="YlGnBu",
                        labels=dict(color="% of listings"))
        fig.update_layout(height=900); st.plotly_chart(fig, use_container_width=True)
        st.subheader("Q10 · Price trend – top 5 most expensive localities")
        top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index.tolist()
        tr = (df[df["Locality"].isin(top5)].groupby(["Year_Built", "Locality"])["Price_in_Lakhs"].mean()
              .unstack("Locality").rolling(5, min_periods=1, center=True).mean()
              .reset_index().melt("Year_Built", var_name="Locality", value_name="Avg price (₹ L)"))
        st.plotly_chart(px.line(tr, x="Year_Built", y="Avg price (₹ L)", color="Locality", markers=True),
                        use_container_width=True)
        bar_note("Prices differ only by a few percent across states/cities, and locality trends are noisy (5-year rolling mean by year built).")

    # ---------------- 11-15
    with t3:
        st.subheader("Q11 · Correlation of numeric features")
        cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_calc", "Age_of_Property", "Floor_No",
                "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Parking_Space", "Security",
                "Amenities_Count", "Infra_Score"]
        fig = px.imshow(df[cols].corr(), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
        fig.update_layout(height=700); st.plotly_chart(fig, use_container_width=True)
        st.subheader("Q12 & Q13 · Schools / hospitals vs price per sq ft")
        a, b = st.columns(2)
        for col, box in [("Nearby_Schools", a), ("Nearby_Hospitals", b)]:
            g = df.groupby(col)["Price_per_SqFt_calc"].mean().reset_index()
            box.plotly_chart(px.line(g, x=col, y="Price_per_SqFt_calc", markers=True), use_container_width=True)
            box.metric(f"Correlation {col} ↔ ₹/sq ft", f"{df[col].corr(df['Price_per_SqFt_calc']):.3f}")
        st.subheader("Q14 · Price by furnished status")
        g = df.groupby("Furnished_Status")[["Price_in_Lakhs", "Price_per_SqFt_calc"]].mean().reset_index()
        a, b = st.columns(2)
        a.plotly_chart(px.bar(g, x="Furnished_Status", y="Price_in_Lakhs"), use_container_width=True)
        b.plotly_chart(px.bar(g, x="Furnished_Status", y="Price_per_SqFt_calc"), use_container_width=True)
        st.subheader("Q15 · Price per sq ft by facing direction")
        g = df.groupby("Facing")["Price_per_SqFt_calc"].mean().sort_values().reset_index()
        st.plotly_chart(px.bar(g, x="Facing", y="Price_per_SqFt_calc"), use_container_width=True)
        bar_note("Apart from the mechanical size/price-per-sq-ft link, correlations are close to zero.")

    # ---------------- 16-20
    with t4:
        st.subheader("Q16 & Q17 · Owner type and availability status")
        a, b = st.columns(2)
        for col, box in [("Owner_Type", a), ("Availability_Status", b)]:
            g = df[col].value_counts().reset_index()
            g.columns = [col, "count"]
            box.plotly_chart(px.bar(g, x=col, y="count", text="count"), use_container_width=True)
        st.subheader("Q18 · Does parking affect price?")
        g = df.assign(Parking=df["Parking_Space"].map({1: "Parking", 0: "No parking"})) \
              .groupby("Parking")[["Price_in_Lakhs", "Price_per_SqFt_calc"]].mean().reset_index()
        a, b = st.columns(2)
        a.plotly_chart(px.bar(g, x="Parking", y="Price_in_Lakhs"), use_container_width=True)
        b.plotly_chart(px.bar(g, x="Parking", y="Price_per_SqFt_calc"), use_container_width=True)
        gp = df.groupby("Parking_Space")["Price_in_Lakhs"].mean()
        st.metric("Price premium with parking", f"{(gp.get(1, np.nan) / gp.get(0, np.nan) - 1) * 100:.2f}%")
        st.subheader("Q19 · Amenities vs price per sq ft")
        a, b = st.columns(2)
        g = df.groupby("Amenities_Count")["Price_per_SqFt_calc"].mean().reset_index()
        a.plotly_chart(px.bar(g, x="Amenities_Count", y="Price_per_SqFt_calc"), use_container_width=True)
        eff = {am: (df.loc[df[f"Has_{am}"] == 1, "Price_per_SqFt_calc"].mean()
                    / df.loc[df[f"Has_{am}"] == 0, "Price_per_SqFt_calc"].mean() - 1) * 100
               for am in L.AMENITY_LIST}
        e = pd.Series(eff).sort_values().reset_index()
        e.columns = ["Amenity", "% difference in ₹/sq ft"]
        b.plotly_chart(px.bar(e, x="% difference in ₹/sq ft", y="Amenity", orientation="h"), use_container_width=True)
        st.subheader("Q20 · Public transport vs price per sq ft and investment potential")
        order = ["Low", "Medium", "High"]
        g = df.groupby("Public_Transport_Accessibility").agg(ppsf=("Price_per_SqFt_calc", "mean"),
                                                             good=("Good_Investment", "mean")).reset_index()
        a, b = st.columns(2)
        a.plotly_chart(px.bar(g, x="Public_Transport_Accessibility", y="ppsf",
                              category_orders={"Public_Transport_Accessibility": order}), use_container_width=True)
        fig = px.bar(g, x="Public_Transport_Accessibility", y="good",
                     category_orders={"Public_Transport_Accessibility": order}, title="Share rated Good Investment")
        fig.update_layout(yaxis_tickformat=".0%"); b.plotly_chart(fig, use_container_width=True)
        bar_note("Transport only matters for the Good-Investment share because it is part of the rule-based appreciation formula.")

    # ---------------- extras
    with t5:
        st.subheader("Area vs estimated 5-year return")
        bins = pd.cut(df["Size_in_SqFt"], bins=[500, 1000, 2000, 3000, 4000, 5000], include_lowest=True)
        g = df.groupby(bins, observed=True)["Return_5Y_Pct"].mean().reset_index()
        g["Size_in_SqFt"] = g["Size_in_SqFt"].astype(str)
        st.plotly_chart(px.bar(g, x="Size_in_SqFt", y="Return_5Y_Pct", labels={"Size_in_SqFt": "Size band (sq ft)"}),
                        use_container_width=True)
        st.subheader("Crime rate vs Good Investment")
        st.warning("The dataset has no crime-rate column, so this analysis from the brief cannot be done. "
                   "It is listed as a limitation / future extension.")
        st.subheader("Infrastructure score vs estimated 5-year price")
        b = pd.cut(df["Infra_Score"], 8)
        g = df.groupby(b, observed=True)["Future_Price_5Y"].mean().reset_index()
        g["Infra_Score"] = g["Infra_Score"].astype(str)
        st.plotly_chart(px.bar(g, x="Infra_Score", y="Future_Price_5Y"), use_container_width=True)
        st.subheader("Good vs not-good investments")
        cmp = df.groupby("Good_Investment")[["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_calc",
                                             "Age_of_Property", "Infra_Score", "Est_Annual_Growth_Pct",
                                             "Return_5Y_Pct"]].mean().T
        cmp.columns = ["Not good (0)", "Good (1)"]
        st.dataframe(cmp.round(2), use_container_width=True)


def page_advisor():
    st.title("🧮 Investment Advisor")
    st.caption("Enter property details. The verdict and 5-year price come from transparent business rules, not a model.")
    cities = sorted(ref["city_median_ppsf"])
    cat = ref["categories"]
    with st.form("prop"):
        c1, c2, c3 = st.columns(3)
        city = c1.selectbox("City", cities)
        c1.caption(f"State: {ref['city_to_state'].get(city, '-')}")
        ptype = c1.selectbox("Property type", cat["Property_Type"])
        bhk = c1.number_input("BHK", 1, 10, 3)
        size = c1.number_input("Size (sq ft)", 100, 20000, int(df["Size_in_SqFt"].median()), step=50)
        price = c2.number_input("Price (₹ lakhs)", 1.0, 5000.0, float(round(df["Price_in_Lakhs"].median(), 1)), step=5.0)
        year = c2.number_input("Year built", 1900, L.CURRENT_YEAR + 5, int(df["Year_Built"].median()))
        furn = c2.selectbox("Furnished status", cat["Furnished_Status"])
        floor = c2.number_input("Floor no.", 0, 100, 5)
        total_floors = c2.number_input("Total floors", 1, 100, 15)
        schools = c3.number_input("Nearby schools", 0, 50, int(df["Nearby_Schools"].median()))
        hospitals = c3.number_input("Nearby hospitals", 0, 50, int(df["Nearby_Hospitals"].median()))
        transport = c3.selectbox("Public transport accessibility", cat["Public_Transport_Accessibility"],
                                 index=min(1, len(cat["Public_Transport_Accessibility"]) - 1))
        availability = c3.selectbox("Availability status", cat["Availability_Status"])
        c4, c5, c6 = st.columns(3)
        parking = c4.checkbox("Parking space", True)
        security = c4.checkbox("Security", True)
        facing = c5.selectbox("Facing", cat["Facing"])
        owner = c5.selectbox("Owner type", cat["Owner_Type"])
        amenities = c6.multiselect("Amenities", L.AMENITY_LIST, default=["Gym"])
        go_ = st.form_submit_button("Evaluate property", type="primary")

    if not go_:
        st.info("Fill the form and press **Evaluate property**.")
        return

    r = L.evaluate_property(dict(city=city, size=size, price=price, year_built=year, schools=schools,
                                 hospitals=hospitals, transport=transport, availability=availability,
                                 security=security, parking=parking), ref)

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        if r["good"]:
            st.success("### ✅ Good Investment")
        else:
            st.error("### ❌ Not a Good Investment")
        st.metric("Estimated price after 5 years", fmt_inr(r["future_price"]), f"+{r['return_pct']:.1f}% ({fmt_inr(r['gain'])})")
        st.metric("Estimated annual appreciation", f"{r['growth']:.2f}% / yr")
    with right:
        st.subheader("Why?  (two rules must both pass)")
        a, b = st.columns(2)
        a.metric("Rule 1 · Price vs city median ₹/sq ft",
                 f"{r['ratio']:.2f}×", "pass ≤ 1.00×" if r["price_ok"] else "fail > 1.00×",
                 delta_color="normal" if r["price_ok"] else "inverse")
        a.caption(f"Your price: ₹ {r['ppsf']:,.0f}/sq ft · {city} median: ₹ {r['city_median_ppsf']:,.0f}/sq ft")
        b.metric("Rule 2 · Annual growth vs dataset median",
                 f"{r['growth']:.2f}%", f"{r['growth'] - r['growth_threshold']:+.2f} pts vs {r['growth_threshold']:.2f}%",
                 delta_color="normal" if r["growth_ok"] else "inverse")
        if not r["price_ok"]:
            st.warning(f"To meet Rule 1 the price would need to be at most about {fmt_inr(r['max_price_for_value'])} for this size.")

    a, b = st.columns(2)
    with a:
        yrs = list(range(0, L.YEARS + 1))
        vals = [price * (1 + r["growth"] / 100) ** y for y in yrs]
        fig = go.Figure(go.Scatter(x=[L.CURRENT_YEAR + y for y in yrs], y=vals, mode="lines+markers"))
        fig.update_layout(title="Projected price (₹ lakhs)", xaxis_title="Year", yaxis_title="₹ lakhs", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        cont = pd.DataFrame({"Factor": list(r["contributions"]), "pts": list(r["contributions"].values())})
        fig = px.bar(cont, x="pts", y="Factor", orientation="h", title="What drives the growth estimate (% / yr)")
        fig.update_layout(height=380, yaxis=dict(autorange="reversed")); st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Comparison with similar listings in {city}")
    peers = df[(df["City"] == city) & (df["BHK"] == bhk)]
    if len(peers):
        c = st.columns(4)
        c[0].metric("Similar listings", f"{len(peers):,}")
        c[1].metric("Avg price", fmt_inr(peers["Price_in_Lakhs"].mean()))
        c[2].metric("Avg ₹/sq ft", f"₹ {peers['Price_per_SqFt_calc'].mean():,.0f}")
        c[3].metric("Good-investment share", f"{peers['Good_Investment'].mean():.1%}")
        st.caption(f"Furnishing ({furn}), facing ({facing}), owner ({owner}), property type ({ptype}), floor and amenities "
                   "are recorded but do not enter the business rules, because they showed almost no effect in the EDA.")
    else:
        st.write("No listings with this city and BHK in the dataset.")


def page_explorer():
    st.title("🗂️ Property Explorer")
    with st.sidebar.expander("Filters", expanded=True):
        states = st.multiselect("State", sorted(df["State"].unique()))
        pool = df[df["State"].isin(states)] if states else df
        cities = st.multiselect("City", sorted(pool["City"].unique()))
        ptypes = st.multiselect("Property type", sorted(df["Property_Type"].unique()))
        bhks = st.multiselect("BHK", sorted(df["BHK"].unique()))
        avail = st.multiselect("Availability", sorted(df["Availability_Status"].unique()))
        furn = st.multiselect("Furnished status", sorted(df["Furnished_Status"].unique()))
        pr = st.slider("Price (₹ lakhs)", float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max()),
                       (float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())))
        sz = st.slider("Size (sq ft)", int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max()),
                       (int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())))
        only_good = st.checkbox("Only Good Investments")

    m = df["Price_in_Lakhs"].between(*pr) & df["Size_in_SqFt"].between(*sz)
    for col, vals in [("State", states), ("City", cities), ("Property_Type", ptypes), ("BHK", bhks),
                      ("Availability_Status", avail), ("Furnished_Status", furn)]:
        if vals:
            m &= df[col].isin(vals)
    if only_good:
        m &= df["Good_Investment"] == 1
    f = df[m]

    if f.empty:
        st.warning("No properties match these filters.")
        return
    c = st.columns(4)
    c[0].metric("Matching properties", f"{len(f):,}")
    c[1].metric("Avg price", fmt_inr(f["Price_in_Lakhs"].mean()))
    c[2].metric("Avg ₹/sq ft", f"₹ {f['Price_per_SqFt_calc'].mean():,.0f}")
    c[3].metric("Good Investment share", f"{f['Good_Investment'].mean():.1%}")

    a, b = st.columns(2)
    g = f.groupby("City")["Price_per_SqFt_calc"].mean().sort_values(ascending=False).head(15).reset_index()
    fig = px.bar(g, x="Price_per_SqFt_calc", y="City", orientation="h", title="Avg ₹/sq ft by city (top 15)")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420); a.plotly_chart(fig, use_container_width=True)
    heat = f.pivot_table(index="City", columns="BHK", values="Price_in_Lakhs", aggfunc="mean")
    if heat.shape[0] > 30:
        heat = heat.loc[f.groupby("City")["Price_in_Lakhs"].mean().nlargest(30).index]
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd", title="Avg price (₹ L): city × BHK")
    fig.update_layout(height=420); b.plotly_chart(fig, use_container_width=True)

    sort_by = st.selectbox("Sort by", ["Return_5Y_Pct", "Price_in_Lakhs", "Price_per_SqFt_calc", "Size_in_SqFt",
                                       "Est_Annual_Growth_Pct"])
    show = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt_calc", "Age_of_Property", "Availability_Status", "Est_Annual_Growth_Pct",
            "Future_Price_5Y", "Return_5Y_Pct", "Good_Investment"]
    view = f.sort_values(sort_by, ascending=False)[show]
    st.dataframe(view.head(500).round(2), use_container_width=True, height=420)
    st.caption("Showing the top 500 rows; download contains every matching row.")
    st.download_button("⬇️ Download filtered properties (CSV)", to_csv_bytes(view), "filtered_properties.csv", "text/csv")


def page_method():
    st.title("ℹ️ Method & Limitations")
    st.markdown(f"""
### No machine learning
This app deliberately uses **no classification or regression models**. Both outputs are computed from explicit rules,
so every result can be explained and edited (constants live in `logic.py`).

### Estimated annual appreciation
`growth % = {L.BASE_GROWTH} + {L.W_INFRA}·Infra_Score + {L.W_UNDER_CONSTRUCTION}·[Under construction] + {L.W_SECURITY}·Security + {L.W_PARKING}·Parking − {abs(L.W_AGE)}·Age`

`Infra_Score = 0.35·school density + 0.35·hospital density + 0.30·transport score` (all scaled 0–1)

### Estimated price after 5 years
`Future price = Price × (1 + growth/100)^5`

### Good Investment
Both must hold:
1. price per sq ft **≤ the city's median** price per sq ft, and
2. estimated annual growth **≥ the dataset median** ({ref['growth_threshold']:.2f}% / yr).

### Limitations
* The dataset has no “good investment” label and no future price, so both are **assumption-based**, not observed market outcomes.
* The EDA shows near-zero relationships between location, amenities and price, which suggests the data is **synthetic**; treat results as a demonstration, not market advice.
* No crime-rate column exists, so that analysis from the brief is not possible.
* The brief's model confidence scores, feature importance and MLflow tracking are not included because they depend on ML models.
""")


{"📊 Overview": page_overview, "🧹 Data & Preprocessing": page_data, "🔎 EDA (20 questions)": page_eda,
 "🧮 Investment Advisor": page_advisor, "🗂️ Property Explorer": page_explorer,
 "ℹ️ Method & Limitations": page_method}[page]()