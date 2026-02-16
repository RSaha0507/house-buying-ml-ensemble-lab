from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# ---- custom models ----
class KMeansLabelModel:
    def __init__(self, n_clusters=6, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y, n_classes=3):
        self.model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=20, random_state=self.random_state)
        labels = self.model.fit_predict(X)
        self.n_classes = n_classes
        self.cluster_proba = np.zeros((self.n_clusters, n_classes), dtype=float)
        for c in range(self.n_clusters):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                self.cluster_proba[c] = np.ones(n_classes) / n_classes
            else:
                counts = np.bincount(y[idx], minlength=n_classes).astype(float)
                self.cluster_proba[c] = (counts + 1.0) / (counts.sum() + n_classes)
        return self

    def predict_proba(self, X):
        nearest = self.model.predict(X)
        return self.cluster_proba[nearest]


class KMedoidsLabelModel:
    def __init__(self, n_clusters=6, random_state=42, max_iter=30):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y, n_classes=3):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        k = min(self.n_clusters, n)
        medoid_idx = rng.choice(n, size=k, replace=False)

        for _ in range(self.max_iter):
            d = ((X[:, None, :] - X[medoid_idx][None, :, :]) ** 2).sum(axis=2)
            assign = d.argmin(axis=1)
            new_medoid_idx = medoid_idx.copy()
            changed = False
            for c in range(k):
                pts = np.where(assign == c)[0]
                if len(pts) == 0:
                    continue
                sub = X[pts]
                dist_sum = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2).sum(axis=1)
                best_local = pts[dist_sum.argmin()]
                if best_local != medoid_idx[c]:
                    changed = True
                    new_medoid_idx[c] = best_local
            medoid_idx = new_medoid_idx
            if not changed:
                break

        self.medoids_ = X[medoid_idx]
        d = ((X[:, None, :] - self.medoids_[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)

        self.n_classes = n_classes
        self.cluster_proba = np.zeros((k, n_classes), dtype=float)
        for c in range(k):
            idx = np.where(assign == c)[0]
            if len(idx) == 0:
                self.cluster_proba[c] = np.ones(n_classes) / n_classes
            else:
                counts = np.bincount(y[idx], minlength=n_classes).astype(float)
                self.cluster_proba[c] = (counts + 1.0) / (counts.sum() + n_classes)
        return self

    def predict_proba(self, X):
        d = ((X[:, None, :] - self.medoids_[None, :, :]) ** 2).sum(axis=2)
        nearest = d.argmin(axis=1)
        return self.cluster_proba[nearest]


# ---- data ----
base = Path(r"C:\Users\91960\house_pricing_nn")
csv_dir = base / "csv"
train_df = pd.read_csv(csv_dir / 'house_buy_train.csv')
cv_df = pd.read_csv(csv_dir / 'house_buy_cv.csv')
test_df = pd.read_csv(csv_dir / 'house_buy_test.csv')

feature_cols = [
    'buyer_income_lpa','house_price_lakh','loan_eligibility','credit_score',
    'down_payment_percent','existing_emi_lpa','employment_years','dependents',
    'property_location_score','employment_type'
]
label_to_int = {'no':0,'neutral':1,'yes':2}

X_train_raw = train_df[feature_cols]
y_train = train_df['can_buy'].map(label_to_int).values
X_cv_raw = cv_df[feature_cols]
y_cv = cv_df['can_buy'].map(label_to_int).values
X_test_raw = test_df[feature_cols]
y_test = test_df['can_buy'].map(label_to_int).values

pre = ColumnTransformer([
    ('num', StandardScaler(), ['buyer_income_lpa','house_price_lakh','credit_score','down_payment_percent','existing_emi_lpa','employment_years','dependents','property_location_score']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['loan_eligibility','employment_type'])
])

X_train = pre.fit_transform(X_train_raw)
X_cv = pre.transform(X_cv_raw)
X_test = pre.transform(X_test_raw)

# dense arrays for custom models
X_train_d = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_cv_d = X_cv.toarray() if hasattr(X_cv, 'toarray') else X_cv
X_test_d = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

# baseline deep NN
nn = MLPClassifier(hidden_layer_sizes=(32,24,12), alpha=0.008, learning_rate_init=0.0015, max_iter=3000, early_stopping=True, validation_fraction=0.2, n_iter_no_change=25, random_state=42)
nn.fit(X_train, y_train)
for nm,X,y in [('train',X_train,y_train),('cv',X_cv,y_cv),('test',X_test,y_test)]:
    p=nn.predict(X)
    acc=accuracy_score(y,p)
    f1=precision_recall_fscore_support(y,p,average='macro',zero_division=0)[2]
    print('baseline_nn',nm,round(acc,4),round(f1,4))

# bootstrap ensemble
B=16
rng=np.random.default_rng(42)

splits = {
    'train': (X_train, X_train_d, y_train),
    'cv': (X_cv, X_cv_d, y_cv),
    'test': (X_test, X_test_d, y_test)
}

proba_sums = {k: np.zeros((v[2].shape[0], 3), dtype=float) for k,v in splits.items()}
model_count = 0

for b in range(B):
    idx = rng.integers(0, len(y_train), size=len(y_train))
    Xb_sparse = X_train[idx]
    Xb_dense = X_train_d[idx]
    yb = y_train[idx]

    learners = [
        ('nn_small', MLPClassifier(hidden_layer_sizes=(8,), alpha=0.01, learning_rate_init=0.0008, max_iter=1200, random_state=100+b, early_stopping=True, validation_fraction=0.2, n_iter_no_change=20), 'sparse'),
        ('nn_deep', MLPClassifier(hidden_layer_sizes=(28,14), alpha=0.005, learning_rate_init=0.002, max_iter=1500, random_state=200+b, early_stopping=True, validation_fraction=0.2, n_iter_no_change=20), 'sparse'),
        ('knn', KNeighborsClassifier(n_neighbors=9, weights='distance'), 'sparse'),
        ('kmeanspp', KMeansLabelModel(n_clusters=7, random_state=300+b), 'dense'),
        ('kmedoids', KMedoidsLabelModel(n_clusters=7, random_state=400+b, max_iter=20), 'dense'),
    ]

    for name, model, mode in learners:
        if mode == 'sparse':
            model.fit(Xb_sparse, yb)
        else:
            model.fit(Xb_dense, yb, n_classes=3)

        for split_name, (Xs_sparse, Xs_dense, _) in splits.items():
            if mode == 'sparse':
                proba = model.predict_proba(Xs_sparse)
            else:
                proba = model.predict_proba(Xs_dense)
            proba_sums[split_name] += proba
        model_count += 1

for split_name, (_, _, y_true) in splits.items():
    avg_proba = proba_sums[split_name] / model_count
    y_pred = avg_proba.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]
    print('ensemble', split_name, round(acc,4), round(f1,4))

print('model_count', model_count)
