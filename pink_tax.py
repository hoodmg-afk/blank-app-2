# %% [markdown]
# # Is There a "Pink Tax" in Designer Fashion? (SSENSE data)
# Run top to bottom. Needs: pandas numpy matplotlib scikit-learn statsmodels
# Data: ssense_dataset.csv (Kaggle SSENSE Fashion Dataset) in the same folder.

# %% Setup
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "ssense_dataset.csv"
MIN_N = 3          # min items per side (women/men) for a matched comparison
SAME_TOL = 1.0     # gaps within +/-1% count as "same"
OUT = Path("results"); OUT.mkdir(exist_ok=True)

# %% 1. Load and clean
raw = pd.read_csv(DATA_PATH)
print(raw.shape, list(raw.columns))

df = raw.rename(columns={"description": "name", "price_usd": "price", "type": "gender"})
df = df[["brand", "name", "price", "gender"]].copy()

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df[df["price"] > 0]

# standardize brand spelling (case/whitespace) to the most common version
df["brand"] = df["brand"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
canon = df.groupby(df["brand"].str.lower())["brand"].agg(lambda s: s.value_counts().index[0])
df["brand"] = df["brand"].str.lower().map(canon)

# "womens" contains "men", so test women first
g = df["gender"].astype(str).str.lower()
df["gender"] = np.where(g.str.contains("women"), "women",
                        np.where(g.str.contains("men"), "men", "other"))
df = df[df["gender"] != "other"]

df = df.dropna(subset=["name"]).drop_duplicates(subset=["brand", "name", "price", "gender"])
df["log_price"] = np.log(df["price"])

# %% 2. Tag item category and pink flag from product names
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


def tag_category(name):
    n = str(name).lower()
    for cat, pat in CATEGORY_RULES:   # first match wins
        if re.search(pat, n):
            return cat
    return "other"


df["category"] = df["name"].map(tag_category)
df["pink"] = df["name"].str.contains(r"\b(?:pink|rose|blush|fuchsia)\b", case=False, regex=True)

print(f"rows={len(df):,} brands={df.brand.nunique()} women's share={(df.gender=='women').mean():.0%} "
      f"uncategorized={(df.category=='other').mean():.0%} pink items={df.pink.sum()}")
df.to_csv(OUT / "clean_data.csv", index=False)

# %% 3. Descriptive overview: raw gap
w_med = df.loc[df.gender == "women", "price"].median()
m_med = df.loc[df.gender == "men", "price"].median()
print(f"Raw median: women ${w_med:.0f} vs men ${m_med:.0f} -> {(w_med / m_med - 1) * 100:+.1f}%")

# %% 4. Q1: matched comparison (same brand + same item type)
d = df[df["category"] != "other"]
cells = d.groupby(["brand", "category", "gender"])["price"].agg(["median", "size"]).unstack("gender")
cells.columns = [f"{stat}_{sex}" for stat, sex in cells.columns]
cells = cells.dropna()
cells = cells[(cells["size_women"] >= MIN_N) & (cells["size_men"] >= MIN_N)].reset_index()
cells["gap_pct"] = (cells["median_women"] / cells["median_men"] - 1) * 100
cells["direction"] = np.select([cells.gap_pct > SAME_TOL, cells.gap_pct < -SAME_TOL],
                               ["women_more", "men_more"], default="same")
cells.to_csv(OUT / "matched_cells.csv", index=False)

print(f"Matched brand+item pairs: {len(cells)}")
print(f"Median matched gap: {cells.gap_pct.median():+.1f}%")
print(cells.direction.value_counts(normalize=True).mul(100).round(1))

by_cat = (cells.groupby("category")["gap_pct"].agg(["median", "size"])
               .rename(columns={"median": "median_gap_pct", "size": "n_pairs"})
               .sort_values("median_gap_pct", ascending=False))
print(by_cat.round(1))

# %% 5. Q2: gap by brand
by_brand = (cells.groupby("brand")
                 .agg(median_gap_pct=("gap_pct", "median"), n_pairs=("gap_pct", "size"),
                      n_women=("size_women", "sum"), n_men=("size_men", "sum"))
                 .sort_values("median_gap_pct", ascending=False))
by_brand.to_csv(OUT / "gap_by_brand.csv")
reliable = by_brand[by_brand.n_pairs >= 3]    # avoid brands with 1-2 pairs
print("Women's premium (3+ pairs):\n", reliable.head(6).round(1))
print("Men's premium (3+ pairs):\n", reliable.tail(6).round(1))

# %% 6. Q3: pink vs non-pink within brand + category + gender
pk = d.groupby(["brand", "category", "gender", "pink"])["price"].agg(["median", "size"]).unstack("pink")
pk.columns = [f"{stat}_{'pink' if p else 'other'}" for stat, p in pk.columns]
pk = pk.dropna()
pk = pk[(pk["size_pink"] >= 2) & (pk["size_other"] >= 2)].reset_index()
pk["gap_pct"] = (pk["median_pink"] / pk["median_other"] - 1) * 100
pk["direction"] = np.select([pk.gap_pct > SAME_TOL, pk.gap_pct < -SAME_TOL],
                            ["pink_more", "pink_less"], default="same")
pk.to_csv(OUT / "pink_cells.csv", index=False)
print(f"Pink comparisons: {len(pk)} | median gap {pk.gap_pct.median():+.1f}%")
print(pk.direction.value_counts(normalize=True).mul(100).round(1))

# %% 7. Goal 2: regression check (log price ~ gender + category + brand)
try:
    import statsmodels.formula.api as smf
    r = d.assign(is_women=(d.gender == "women").astype(int))
    fit = smf.ols("log_price ~ is_women + C(category) + C(brand)", data=r).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(r["brand"])[0]})
    b, (lo, hi) = fit.params["is_women"], fit.conf_int().loc["is_women"]
    pct = lambda x: (np.exp(x) - 1) * 100
    print(f"Women's premium holding brand+category fixed: {pct(b):+.1f}% "
          f"(95% CI {pct(lo):+.1f}% to {pct(hi):+.1f}%), p={fit.pvalues['is_women']:.3f}")
except ImportError:
    print("Install statsmodels to run the regression check: pip install statsmodels")

# %% 8. Q4: gender classifier (does price add information?)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_model(use_price, use_name):
    parts = [("cat", OneHotEncoder(handle_unknown="ignore"), ["brand", "category"])]
    if use_price:
        parts.append(("price", StandardScaler(), ["log_price"]))
    if use_name:
        parts.append(("name", TfidfVectorizer(min_df=5, ngram_range=(1, 2)), "name"))
    return Pipeline([("prep", ColumnTransformer(parts)), ("clf", LogisticRegression(max_iter=2000))])


y = (df["gender"] == "women").astype(int)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for label, (p, n) in {"brand+category": (False, False), "brand+category+price": (True, False),
                      "brand+category+name": (False, True), "all features": (True, True)}.items():
    res = cross_validate(make_model(p, n), df, y, cv=cv, scoring=["roc_auc", "accuracy"])
    rows.append({"features": label, "auc": res["test_roc_auc"].mean(), "accuracy": res["test_accuracy"].mean()})
cv_table = pd.DataFrame(rows)
cv_table.to_csv(OUT / "classifier_cv.csv", index=False)
print(cv_table.round(3))
# Caution: product names often contain gendered words ("miniskirt", "blouse"), so the
# name-based scores partly reflect the label itself.

# %% 9. Charts for the slides
PINK, BLUE, BG, TEXT = "#c2185b", "#2f6fba", "#f4efee", "#7a1427"


def gap_chart(series, title, fname):
    s = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(s.index, s.values, color=[PINK if v > 0 else BLUE for v in s.values])
    ax.axvline(0, color="#999", lw=0.8)
    pad = s.abs().max() * 0.02
    for y_, v in enumerate(s.values):
        ax.text(v + (pad if v >= 0 else -pad), y_, f"{v:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", color=TEXT, fontsize=10)
    lim = s.abs().max() * 1.2
    ax.set_xlim(-lim, lim); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=TEXT, fontsize=13, pad=26)
    ax.text(0, 1.01, "◀ Men's cost more", transform=ax.transAxes, color=BLUE, fontsize=9)
    ax.text(1, 1.01, "Women's cost more ▶", transform=ax.transAxes, color=PINK, fontsize=9, ha="right")
    ax.tick_params(axis="y", length=0, labelcolor=TEXT)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200, facecolor=BG); plt.close(fig)


gap_chart(by_cat[by_cat.n_pairs >= 5]["median_gap_pct"], "Median gap by item type", "gap_by_item_type.png")
ends = pd.concat([reliable.head(6), reliable.tail(6)])
gap_chart(ends["median_gap_pct"], "Median gap by brand (3+ matched item types)", "gap_by_brand.png")
print("Saved tables and charts in", OUT.resolve())
