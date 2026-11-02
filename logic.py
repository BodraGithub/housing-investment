"""Data cleaning, feature engineering and rule-based investment logic.

No machine learning is used anywhere. The "Good Investment" label and the
5-year price estimate come from the transparent business rules defined in the
EDA notebook (Part 4).
"""
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025
YEARS = 5

# ---- assumptions (same as the notebook; edit to change the rules) ----
BASE_GROWTH = 5.0
W_INFRA = 3.0
W_UNDER_CONSTRUCTION = 1.0
W_SECURITY = 0.5
W_PARKING = 0.5
W_AGE = -0.04

AMENITY_LIST = ["Gym", "Pool", "Clubhouse", "Playground", "Garden"]
TRANSPORT_SCORE = {"low": 1 / 3, "medium": 2 / 3, "high": 1.0}


def iqr_bounds(s: pd.Series):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _yes_no(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["yes", "y", "1", "true"]).astype(int)


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0.0


def estimate_growth(infra, under_construction, security, parking, age):
    """Estimated annual appreciation in % per year (works on scalars or Series)."""
    return (BASE_GROWTH + W_INFRA * infra + W_UNDER_CONSTRUCTION * under_construction
            + W_SECURITY * security + W_PARKING * parking + W_AGE * age)


def prepare(raw: pd.DataFrame):
    """Clean + engineer features. Returns (df, ref, quality)."""
    quality = {
        "rows_raw": len(raw),
        "columns_raw": raw.shape[1],
        "missing_total": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.drop(columns=["ID"], errors="ignore").duplicated().sum()),
        "floor_gt_total": float((raw["Floor_No"] > raw["Total_Floors"]).mean()),
        "ppsf_zero_after_rounding": float((raw["Price_per_SqFt"] == 0).mean()),
    }

    df = raw.copy()
    # defensive cleaning
    subset = [c for c in df.columns if c != "ID"]
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "ID"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in df.select_dtypes(exclude=np.number).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    # floors
    df["Floor_Data_Inconsistent"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)
    df["Floor_No"] = np.minimum(df["Floor_No"], df["Total_Floors"])

    # age and price per sq ft (Rs per sq ft)
    df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
    df["Price_per_SqFt_calc"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    lo, hi = iqr_bounds(df["Price_per_SqFt_calc"])
    df["PPSF_Outlier_Flag"] = ((df["Price_per_SqFt_calc"] < lo) | (df["Price_per_SqFt_calc"] > hi)).astype(int)

    # binary encodings
    df["Parking_Space"] = _yes_no(df["Parking_Space"])
    df["Security"] = _yes_no(df["Security"])

    # engineered features
    df["Amenities_Count"] = df["Amenities"].str.split(",").str.len()
    for a in AMENITY_LIST:
        df[f"Has_{a}"] = df["Amenities"].str.contains(a, case=False).astype(int)

    s_min, s_max = float(df["Nearby_Schools"].min()), float(df["Nearby_Schools"].max())
    h_min, h_max = float(df["Nearby_Hospitals"].min()), float(df["Nearby_Hospitals"].max())
    df["School_Density_Score"] = _minmax(df["Nearby_Schools"])
    df["Hospital_Density_Score"] = _minmax(df["Nearby_Hospitals"])
    df["Transport_Score"] = df["Public_Transport_Accessibility"].str.lower().map(TRANSPORT_SCORE).fillna(0.5)
    df["Infra_Score"] = (0.35 * df["School_Density_Score"] + 0.35 * df["Hospital_Density_Score"]
                         + 0.30 * df["Transport_Score"])
    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"]
    df["Size_per_BHK"] = df["Size_in_SqFt"] / df["BHK"]

    city_median_ppsf = df.groupby("City")["Price_per_SqFt_calc"].median()
    df["City_Median_PPSF"] = df["City"].map(city_median_ppsf)
    df["Price_vs_City_Median"] = df["Price_per_SqFt_calc"] / df["City_Median_PPSF"]

    # rule-based targets
    df["Under_Construction"] = df["Availability_Status"].str.lower().str.contains("under").astype(int)
    df["Est_Annual_Growth_Pct"] = estimate_growth(df["Infra_Score"], df["Under_Construction"],
                                                  df["Security"], df["Parking_Space"], df["Age_of_Property"])
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Est_Annual_Growth_Pct"] / 100) ** YEARS
    df["Return_5Y_Pct"] = (df["Future_Price_5Y"] / df["Price_in_Lakhs"] - 1) * 100
    growth_threshold = float(df["Est_Annual_Growth_Pct"].median())
    df["Good_Investment"] = ((df["Price_vs_City_Median"] <= 1.0)
                             & (df["Est_Annual_Growth_Pct"] >= growth_threshold)).astype(int)

    ref = {
        "school_min": s_min, "school_max": s_max, "hosp_min": h_min, "hosp_max": h_max,
        "city_median_ppsf": city_median_ppsf.to_dict(),
        "city_to_state": df.drop_duplicates("City").set_index("City")["State"].to_dict(),
        "growth_threshold": growth_threshold,
        "categories": {c: sorted(df[c].unique().tolist()) for c in
                       ["Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
                        "Facing", "Owner_Type", "Availability_Status"]},
        "year_min": int(df["Year_Built"].min()), "year_max": int(df["Year_Built"].max()),
        "size_min": int(df["Size_in_SqFt"].min()), "size_max": int(df["Size_in_SqFt"].max()),
        "price_min": float(df["Price_in_Lakhs"].min()), "price_max": float(df["Price_in_Lakhs"].max()),
        "outlier_bounds_ppsf": (float(lo), float(hi)),
    }
    quality["ppsf_outliers"] = int(df["PPSF_Outlier_Flag"].sum())
    quality["rows_clean"] = len(df)
    return df, ref, quality


def evaluate_property(p: dict, ref: dict) -> dict:
    """Apply the business rules to one property entered in the form."""
    school_rng = max(ref["school_max"] - ref["school_min"], 1)
    hosp_rng = max(ref["hosp_max"] - ref["hosp_min"], 1)
    school_score = (p["schools"] - ref["school_min"]) / school_rng
    hosp_score = (p["hospitals"] - ref["hosp_min"]) / hosp_rng
    transport = TRANSPORT_SCORE.get(p["transport"].lower(), 0.5)
    infra = 0.35 * school_score + 0.35 * hosp_score + 0.30 * transport

    age = CURRENT_YEAR - p["year_built"]
    uc = int("under" in p["availability"].lower())
    growth = estimate_growth(infra, uc, int(p["security"]), int(p["parking"]), age)
    future = p["price"] * (1 + growth / 100) ** YEARS

    ppsf = p["price"] * 1e5 / p["size"]
    city_med = ref["city_median_ppsf"].get(p["city"], np.nan)
    ratio = ppsf / city_med if city_med else np.nan
    price_ok = bool(ratio <= 1.0)
    growth_ok = bool(growth >= ref["growth_threshold"])

    contributions = {
        "Base growth": BASE_GROWTH,
        "Infrastructure (schools, hospitals, transport)": W_INFRA * infra,
        "Under construction": W_UNDER_CONSTRUCTION * uc,
        "Security": W_SECURITY * int(p["security"]),
        "Parking": W_PARKING * int(p["parking"]),
        "Age of property": W_AGE * age,
    }
    return {
        "good": price_ok and growth_ok, "price_ok": price_ok, "growth_ok": growth_ok,
        "infra": infra, "growth": growth, "future_price": future,
        "gain": future - p["price"], "return_pct": (future / p["price"] - 1) * 100,
        "ppsf": ppsf, "city_median_ppsf": city_med, "ratio": ratio,
        "max_price_for_value": city_med * p["size"] / 1e5 if city_med == city_med else np.nan,
        "growth_threshold": ref["growth_threshold"], "contributions": contributions, "age": age,
    }