# =========================
# KNN Classifier with PCA Plot
# =========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load basic settings
CSV_PATH = "spotify.csv"
GENRE_COL = "track_genre"
FEATURES = ["tempo", "loudness", "energy"]
TOP_N_GENRES = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 2. Read dataset
df = pd.read_csv(CSV_PATH)
print("CSV loaded. Columns:", df.columns.tolist())

# keep only needed columns and drop missing rows
df = df[FEATURES + [GENRE_COL]].copy()
df = df.dropna()

print("Data after dropping NaNs:", df.shape)

# 3. Use only the top genres
top_genres = df[GENRE_COL].value_counts().nlargest(TOP_N_GENRES).index.tolist()
df = df[df[GENRE_COL].isin(top_genres)].copy()

print("Genres used:", top_genres)

# 4. Set up X and y
X = df[FEATURES].values
y_raw = df[GENRE_COL].values

# encode genres into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print("\nGenre labels:")
for i, genre in enumerate(label_encoder.classes_):
    print(i, "=", genre)

# 5. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# 6. Scale the features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Try different k values
k_values = list(range(1, 21))
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', color='darkblue')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy for different k values")
plt.grid(alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# choose stable k
max_acc = max(accuracies)
stable_k = next(k for k, acc in zip(k_values, accuracies) if acc >= max_acc - 0.01)

print("Best stable k:", stable_k, "  Max accuracy:", max_acc)

# 8. Train the final KNN model
knn_final = KNeighborsClassifier(n_neighbors=stable_k)
knn_final.fit(X_train_scaled, y_train)
y_pred_final = knn_final.predict(X_test_scaled)

# 9. Print accuracy and report
print("Test Accuracy:", accuracy_score(y_test, y_pred_final))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=label_encoder.classes_))

# confusion matrix
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 10. PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(12, 9))

palette = sns.color_palette("tab10", len(np.unique(y_train)))

for genre_label, color in zip(np.unique(y_train), palette):
    idx = y_train == genre_label
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        s=80,
        color=color,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.9,
        label=label_encoder.classes_[genre_label]
    )

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Plot of Songs by Genre")
plt.legend(title="Genres")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
