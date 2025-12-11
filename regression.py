import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset.csv")

# Convert explicit + small ints to efficient types
df["explicit"] = df["explicit"].astype("int8")
df["mode"] = df["mode"].astype("int8")
df["key"] = df["key"].astype("int8")
df["time_signature"] = df["time_signature"].astype("int8")


numeric_features = [
    'danceability','energy','key','loudness','mode','speechiness',
    'acousticness','instrumentalness','liveness','valence','tempo',
    'duration_ms','time_signature','explicit'
]

# Handle genres: drop rare genres
genre_counts = df["track_genre"].value_counts()
common_genres = genre_counts[genre_counts > 100].index
df = df[df["track_genre"].isin(common_genres)]

# One-hot encode remaining common genres
genre_dummies = pd.get_dummies(df["track_genre"], prefix="genre").astype("int8")

# PCA on genre one-hot encoding (reduce dimension!)
pca = PCA(n_components=10, random_state=42)
genre_pca_array = pca.fit_transform(genre_dummies)
genre_pca = pd.DataFrame(genre_pca_array, columns=[f"genre_pc{i}" for i in range(10)])

# INTERPRET PCA COMPONENTS (Automatic naming)
loadings = pd.DataFrame(
    pca.components_.T,
    index=genre_dummies.columns,
    columns=[f"genre_pc{i}" for i in range(10)]
)

def name_pc(pc_name):
    """Return a human-readable description of a PCA component."""
    top_pos = loadings[pc_name].sort_values(ascending=False).head(3)
    top_neg = loadings[pc_name].sort_values().head(3)
    
    pos_genres = [g.replace("genre_", "") for g in top_pos.index]
    neg_genres = [g.replace("genre_", "") for g in top_neg.index]

    return f"{pos_genres} vs {neg_genres}"

# Create mapping: "genre_pcX" -> "genre_pcX (Pop / EDM cluster...)"
pc_name_map = {
    f"genre_pc{i}": name_pc(f"genre_pc{i}") for i in range(10)
}

print("\n=== PCA Genre Component Interpretation ===")
for pc, name in pc_name_map.items():
    print(f"{pc}: {name}")

# Combine numeric + compressed genre features
X = pd.concat([
    df[numeric_features].reset_index(drop=True),
    genre_pca.reset_index(drop=True)
], axis=1)

y = df["popularity"].astype("int16")  # save memory

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = ExtraTreesRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

print("\nExtraTrees R² score:", model.score(X_test, y_test))

importances = model.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Replace genre_pc names with descriptive labels
def rename_feature(f):
    return f"{f} ({pc_name_map[f]})" if f in pc_name_map else f

feat_imp["feature"] = feat_imp["feature"].apply(rename_feature)

print("\nTop 20 Most Important Features:")
print(feat_imp.head(20))

plt.figure(figsize=(10, 8))
sns.barplot(
    data=feat_imp.head(20),
    x="importance",
    y="feature",
    hue="feature",
    legend=False,
    palette="magma"
)
plt.title("Top 20 Feature Importances (Optimized ExtraTrees Model)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
