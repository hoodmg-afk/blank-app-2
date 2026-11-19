"""
Is There a "Pink Tax" in Designer Fashion?  (SSENSE data)
Self-contained Streamlit app: one file + ssense_dataset.csv in the same folder.
Run locally:  streamlit run streamlit_app.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pink Tax in Designer Fashion?", page_icon="👗", layout="wide")

DATA_PATH = Path(__file__).parent / "ssense_dataset.csv"
MIN_N = 3          # min items per side for a matched comparison
SAME_TOL = 1.0     # gaps within +/-1% count as "same"
PINK, BLUE = "#d6336c", "#2f6fba"

CATEGORY_RULES = [
    ("sneakers", r"sneaker|trainer|runner"),
    ("boots", r"\bboots?\b|chelsea"),
    ("sandals", r"sandal|slide|flip[- ]flop"),
    ("loafers", r"loafer|derby|oxford|brogue"),
    ("jeans", r"\bjeans?\b|denim trouser"),
    ("trousers", r"trouser|pants|chino|cargo|jogger|sweatpant"),
    ("shorts", r"\bshorts?\b"),
    ("skirts", r"\bskirt"),
    ("dresses", r"\bdress|gown|jumpsuit"),
    ("t-shirts", r"t-?shirt|\btee\b"),
    ("sweatshirts", r"hoodie|sweatshirt"),
    ("shirts", r"\bshirt|blouse|polo"),
    ("sweaters", r"sweater|cardigan|knit|pullover|jumper"),
    ("coats", r"\bcoat|parka|trench|puffer|overcoat"),
    ("jackets", r"jacket|blazer|bomber|vest|gilet"),
    ("bags", r"\bbag\b|tote|backpack|clutch|satchel|pouch|wallet|purse"),
    ("sunglasses", r"sunglass|glasses|eyewear"),
    ("hats", r"\bhat\b|\bcap\b|beanie"),
    ("belts", r"\bbelt"),
    ("scarves", r"scarf|scarves"),
    ("jewelry", r"necklace|ring\b|earring|bracelet"),
]


def tag_category(name: str) -> str:
    n = str(name).lower()
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, n):
            return cat
    return "other"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Can't find {DATA_PATH.name}. Put it in the same folder as this file.")
        st.stop()
    df = pd.read_csv(DATA_PATH).rename(
        columns={"description": "name", "price_usd": "price", "type": "gender"})
    df = df[["brand", "name", "price", "gender"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0]

    df["brand"] = df["brand"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    canon = df.groupby(df["brand"].str.lower())["brand"].agg(lambda s: s.value_counts().index[0])
    df["brand"] = df["brand"].str.lower().map(canon)

    g = df["gender"].astype(str).str.lower()          # "womens" contains "men": test women first
    df["gender"] = np.where(g.str.contains("women"), "women",
                            np.where(g.str.contains("men"), "men", "other"))
    df = df[df["gender"] != "other"]
    df = df.dropna(subset=["name"]).drop_duplicates(subset=["brand", "name", "price", "gender"])

    df["category"] = df["name"].map(tag_category)
    df["pink"] = df["name"].str.contains(r"\b(?:pink|rose|blush|fuchsia)\b", case=False, regex=True)
    df["log_price"] = np.log(df["price"])
    return df.reset_index(drop=True)


def matched_cells(df: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    d = df[df["category"] != "other"]
    g = d.groupby(["brand", "category", "gender"])["price"].agg(["median", "size"]).unstack("gender")
    if g.empty or not {"women", "men"} <= set(g.columns.get_level_values(1)):
        return pd.DataFrame(columns=["brand", "category", "gap_pct", "direction",
                                     "size_women", "size_men"])
    g.columns = [f"{stat}_{sex}" for stat, sex in g.columns]
    g = g.dropna()
    g = g[(g["size_women"] >= min_n) & (g["size_men"] >= min_n)].reset_index()
    g["gap_pct"] = (g["median_women"] / g["median_men"] - 1) * 100
    g["direction"] = np.select([g.gap_pct > SAME_TOL, g.gap_pct < -SAME_TOL],
                               ["Women's cost more", "Men's cost more"], default="Same")
    return g


def pink_cells(df: pd.DataFrame, min_n: int = 2) -> pd.DataFrame:
    d = df[df["category"] != "other"]
    g = d.groupby(["brand", "category", "gender", "pink"])["price"].agg(["median", "size"]).unstack("pink")
    if g.empty or g.columns.get_level_values(1).nunique() < 2:
        return pd.DataFrame(columns=["brand", "category", "gender", "gap_pct", "direction"])
    g.columns = [f"{stat}_{'pink' if p else 'other'}" for stat, p in g.columns]
    g = g.dropna()
    g = g[(g["size_pink"] >= min_n) & (g["size_other"] >= min_n)].reset_index()
    g["gap_pct"] = (g["median_pink"] / g["median_other"] - 1) * 100
    g["direction"] = np.select([g.gap_pct > SAME_TOL, g.gap_pct < -SAME_TOL],
                               ["Pink costs more", "Pink costs less"], default="Same")
    return g


def brand_gaps(cells: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    out = (cells.groupby("brand")
                .agg(median_gap_pct=("gap_pct", "median"), n_pairs=("gap_pct", "size"),
                     n_women=("size_women", "sum"), n_men=("size_men", "sum")))
    return out[out.n_pairs >= min_pairs].sort_values("median_gap_pct", ascending=False)


def gap_bar(s: pd.Series, xlabel: str):
    d = s.reset_index()
    d.columns = ["label", "gap"]
    d["Direction"] = np.where(d.gap > 0, "Women's cost more", "Men's cost more")
    fig = px.bar(d, x="gap", y="label", orientation="h", color="Direction",
                 color_discrete_map={"Women's cost more": PINK, "Men's cost more": BLUE},
                 labels={"gap": xlabel, "label": ""}, text=d.gap.map(lambda v: f"{v:+.0f}%"))
    fig.update_layout(yaxis=dict(autorange="reversed"), legend_title_text="", height=max(300, 28 * len(d) + 120))
    return fig


# --------------------------------------------------------------------------
# Models (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models...")
def train_models(df: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    d = df[df["category"] != "other"].copy()
    d["is_women"] = (d["gender"] == "women").astype(int)
    clf = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["brand", "category"]),
                                    ("price", StandardScaler(), ["log_price"])])),
        ("clf", LogisticRegression(max_iter=2000))]).fit(d, d["is_women"])
    reg = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["brand", "category"]),
                                    ("g", "passthrough", ["is_women"])])),
        ("reg", Ridge(alpha=1.0))]).fit(d, d["log_price"])
    return clf, reg


@st.cache_data(show_spinner="Running cross-validation (about a minute)...")
def classifier_cv(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    y = (df["gender"] == "women").astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for label, (p, n) in {"Brand + category": (False, False),
                          "Brand + category + price": (True, False),
                          "Brand + category + name words": (False, True),
                          "All features": (True, True)}.items():
        parts = [("cat", OneHotEncoder(handle_unknown="ignore"), ["brand", "category"])]
        if p:
            parts.append(("price", StandardScaler(), ["log_price"]))
        if n:
            parts.append(("name", TfidfVectorizer(min_df=5, ngram_range=(1, 2)), "name"))
        pipe = Pipeline([("prep", ColumnTransformer(parts)), ("clf", LogisticRegression(max_iter=2000))])
        r = cross_validate(pipe, df, y, cv=cv, scoring=["roc_auc", "accuracy"])
        rows.append({"Features": label, "AUC": r["test_roc_auc"].mean(), "Accuracy": r["test_accuracy"].mean()})
    return pd.DataFrame(rows)


@st.cache_data
def regression_check(df: pd.DataFrame):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    d = df[df["category"] != "other"].copy()
    d["is_women"] = (d["gender"] == "women").astype(int)
    fit = smf.ols("log_price ~ is_women + C(category) + C(brand)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["brand"])[0]})
    b, (lo, hi) = fit.params["is_women"], fit.conf_int().loc["is_women"]
    pct = lambda x: (np.exp(x) - 1) * 100
    return pct(b), pct(lo), pct(hi), fit.pvalues["is_women"]


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
df_all = load_data()
st.title("Is There a Pink Tax in Designer Fashion?")
st.caption("SSENSE Fashion Dataset (Kaggle, 2023). Gaps are associations, not proof of discrimination: "
           "we can't see materials or quality, 'women's/men's' is SSENSE's own label, and this is one retailer.")

# Sidebar filters
st.sidebar.header("Filters")
sel_brands = st.sidebar.multiselect("Brand", sorted(df_all.brand.unique()))
sel_cats = st.sidebar.multiselect("Product category", sorted(c for c in df_all.category.unique() if c != "other"))
sel_genders = st.sidebar.multiselect("Gender label", ["women", "men"], default=["women", "men"])
pmin, pmax = int(df_all.price.min()), int(df_all.price.max())
price_rng = st.sidebar.slider("Price range ($)", pmin, pmax, (pmin, pmax))
pink_only = st.sidebar.checkbox("Pink items only")
min_pairs = st.sidebar.slider("Min matched item types per brand", 1, 6, 3,
                              help="Brands with fewer matched item types are hidden from brand charts.")

df = df_all[df_all.gender.isin(sel_genders) & df_all.price.between(*price_rng)]
if sel_brands:
    df = df[df.brand.isin(sel_brands)]
if sel_cats:
    df = df[df.category.isin(sel_cats)]
if pink_only:
    df = df[df.pink]

cells = matched_cells(df)
bg = brand_gaps(cells, min_pairs)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Median matched gap", f"{cells.gap_pct.median():+.1f}%" if len(cells) else "n/a",
          help="Median women's-vs-men's price gap within the same brand and item type")
c2.metric("Brands with women's premium", f"{(bg.median_gap_pct > SAME_TOL).sum()} of {len(bg)}" if len(bg) else "n/a")
c3.metric("Matched brand + item pairs", f"{len(cells):,}")
c4.metric("Items analyzed", f"{len(df):,}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Price Gap by Brand", "Pink vs. Non-Pink", "Models & Predict"])

# ---- Overview ----
with tab1:
    w, m = df_all[df_all.gender == "women"].price.median(), df_all[df_all.gender == "men"].price.median()
    all_cells = matched_cells(df_all)
    a, b = st.columns(2)
    a.subheader("Raw vs matched")
    a.write(f"**Raw gap (all items):** women's median ${w:,.0f} vs men's ${m:,.0f} -> **{(w / m - 1) * 100:+.0f}%**")
    a.write(f"**Matched gap (same brand + item type):** **{all_cells.gap_pct.median():+.1f}%** "
            f"across {len(all_cells)} pairs")
    shares = all_cells.direction.value_counts(normalize=True).mul(100).round(0)
    a.write(" | ".join(f"{k}: {v:.0f}%" for k, v in shares.items()))
    a.caption("Most of the raw gap comes from what each section sells, not the label.")
    if len(cells):
        by_cat = cells.groupby("category")["gap_pct"].agg(["median", "size"])
        by_cat = by_cat[by_cat["size"] >= 5]["median"].sort_values(ascending=False)
        b.subheader("Median gap by item type")
        b.plotly_chart(gap_bar(by_cat, "Median gap (%)"))
        b.caption("Item types with at least 5 matched brand pairs.")
    else:
        b.info("No matched comparisons for these filters.")

# ---- Brand gaps ----
with tab2:
    left, right = st.columns(2)
    if len(bg):
        n_end = left.slider("Brands to show at each end", 3, 15, 6)
        show = pd.concat([bg.head(n_end), bg.tail(n_end)])
        show = show[~show.index.duplicated()]["median_gap_pct"]
        left.plotly_chart(gap_bar(show, "Median gap (%)"))
        left.caption("Each bar is the median across that brand's matched item types.")
    else:
        left.info("No brands meet the minimum for these filters. Lower the slider in the sidebar.")
    cats_avail = sorted(c for c in df.category.unique() if c != "other")
    if cats_avail:
        cat_pick = right.selectbox("Price distribution within one category", cats_avail)
        right.plotly_chart(px.box(df[df.category == cat_pick], x="gender", y="price", log_y=True,
                                  color="gender", color_discrete_map={"women": PINK, "men": BLUE},
                                  labels={"price": "Price ($, log scale)", "gender": ""}))
    if len(bg):
        st.subheader("Brand detail")
        b_pick = st.selectbox("Pick a brand", list(bg.index))
        st.dataframe(cells[cells.brand == b_pick][["category", "median_women", "median_men",
                                                   "size_women", "size_men", "gap_pct"]].round(1))
        st.dataframe(df[df.brand == b_pick][["name", "category", "gender", "price"]]
                     .sort_values("price", ascending=False))

# ---- Pink ----
with tab3:
    pmin_n = st.slider("Min items per side (pink and non-pink)", 1, 4, 2)
    pc = pink_cells(df_all if not (sel_brands or sel_cats) else df, pmin_n)
    if len(pc):
        a, b = st.columns(2)
        a.metric("Median pink vs non-pink gap", f"{pc.gap_pct.median():+.1f}%")
        a.write(f"Based on **{len(pc)}** comparisons within the same brand, category and gender.")
        a.write(pc.direction.value_counts(normalize=True).mul(100).round(0).astype(int).astype(str).add("%"))
        agg = pc.groupby("category")["gap_pct"].median().sort_values(ascending=False)
        b.plotly_chart(gap_bar(agg, "Pink vs non-pink gap (%)"))
        st.caption("A consistent premium would show most bars on the same side. Small samples: be cautious. "
                   "Lowering the minimum adds comparisons but makes each one noisier.")
    else:
        st.info("Not enough pink items under these filters.")

# ---- Models ----
with tab4:
    st.subheader("Does the label still matter once brand and item type are held fixed?")
    reg = regression_check(df_all)
    if reg:
        st.write(f"Regression on log price (brand + category fixed): women's premium **{reg[0]:+.1f}%** "
                 f"(95% CI {reg[1]:+.1f}% to {reg[2]:+.1f}%, p = {reg[3]:.3f}).")
    else:
        st.caption("Install `statsmodels` (add it to requirements.txt) to show the regression check.")

    st.subheader("Can a model guess the label?")
    if st.button("Run 5-fold cross-validation"):
        res = classifier_cv(df_all)
        st.dataframe(res.round(3))
        st.caption("If price barely raises AUC once brand and category are known, price adds little. "
                   "Product names often contain gendered words, so name-based scores partly reflect the label.")

    st.subheader("Try an item")
    clf, reg_model = train_models(df_all)
    d = df_all[df_all.category != "other"]
    x1, x2, x3 = st.columns(3)
    b_in = x1.selectbox("Brand", sorted(d.brand.unique()), key="b_in")
    c_in = x2.selectbox("Category", sorted(d.category.unique()), key="c_in")
    p_in = x3.number_input("Price ($)", 20.0, 15000.0, 400.0, step=10.0)
    if st.button("Predict"):
        row = pd.DataFrame({"brand": [b_in], "category": [c_in], "log_price": [np.log(p_in)]})
        p = clf.predict_proba(row)[0, 1]
        st.write(f"**Predicted label:** {'Women' if p >= 0.5 else 'Men'} ({max(p, 1 - p):.0%} confidence)")
        pr = pd.DataFrame({"brand": [b_in] * 2, "category": [c_in] * 2, "is_women": [1, 0]})
        pw, pm = np.exp(reg_model.predict(pr))
        st.write(f"**Typical price for this brand + category:** women's ${pw:,.0f} vs men's ${pm:,.0f}")
