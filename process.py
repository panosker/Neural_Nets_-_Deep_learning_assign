from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,RocCurveDisplay,roc_curve)
import tensorflow as tf
import keras
from keras import layers, models, regularizers, callbacks
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt
import shap

SEED = 42
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df[df["TotalCharges"].isna()][["customerID", "tenure", "TotalCharges"]])

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("Duplicates:", df.duplicated().sum())
df = df.drop(columns="customerID")


df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
df[service_cols] = df[service_cols].replace({"No internet service": "No",
                                             "No phone service": "No"})


binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"] + service_cols
for c in binary_cols:
    df[c] = df[c].map({"Yes": 1, "No": 0})
df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"],
                    drop_first=True, dtype=int)


X = df.drop(columns="Churn")
y = df["Churn"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=SEED)


num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


print(X_train.shape, X_val.shape, X_test.shape)
print("Churn rate:", y_train.mean().round(3), y_val.mean().round(3), y_test.mean().round(3))
print(X_train[num_cols].describe().loc[["mean", "std"]].round(2))
print(df.isna().sum().sum(), "missing values left")

def evaluate(model, X, y, threshold=0.5):
    """Return the five required metrics for a fitted model."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {"Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred),
            "F1-score": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, proba)}

# ---------- Two baseline variants ----------
# (a) plain Logistic Regression (L2 penalty, C=1 by default)
lr_plain = LogisticRegression(max_iter=1000, random_state=SEED)
lr_plain.fit(X_train, y_train)

# (b) same model with class weights to compensate the 26.5% / 73.5% imbalance
lr_bal = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
lr_bal.fit(X_train, y_train)

# ---------- Compare on the VALIDATION set (test set stays untouched) ----------
val_results = pd.DataFrame({"LR (no weights)": evaluate(lr_plain, X_val, y_val),
                            "LR (balanced)": evaluate(lr_bal, X_val, y_val)}).T.round(3)
print(val_results)

# ---------- Final baseline = chosen variant, evaluated ONCE on the test set ----------
baseline = lr_bal
test_results = pd.Series(evaluate(baseline, X_test, y_test)).round(3)
print("\nTest set results (baseline):\n", test_results)

# ---------- Confusion matrix ----------
pred_test = baseline.predict(X_test)
print(confusion_matrix(y_test, pred_test))
ConfusionMatrixDisplay.from_predictions(y_test, pred_test,
                                        display_labels=["Stayed", "Churned"], cmap="Blues")
plt.title("Logistic Regression: confusion matrix (test)")
plt.savefig("figures/fig4_lr_confusion.png", bbox_inches="tight"); plt.show()

# ---------- ROC curve ----------
RocCurveDisplay.from_estimator(baseline, X_test, y_test)
plt.title("Logistic Regression: ROC curve (test)")
plt.savefig("figures/fig5_lr_roc.png", bbox_inches="tight"); plt.show()

# ---------- Interpretability: odds ratios ----------
odds = pd.Series(np.exp(baseline.coef_[0]), index=X_train.columns).sort_values()
print(odds.round(2))

# Για αναπαραγωγή ίδιων αποτελεσμάτων
tf.random.set_seed(SEED)

# ============================================================
# 3.1 ΑΠΛΟ MLP (ΠΡΩΤΟ ΣΗΜΕΙΟ ΣΥΓΚΡΙΣΗΣ ΓΙΑ ΤΟ ΝΕΥΡΩΝΙΚΟ)
# ============================================================

# ---- Προετοιμασία δεδομένων ----
# Μετατροπή σε float32 (η μορφή που περιμένει το Keras)
Xtr = X_train.values.astype("float32")
Xva = X_val.values.astype("float32")
Xte = X_test.values.astype("float32")
ytr, yva, yte = y_train.values, y_val.values, y_test.values

# Βάρη κλάσεων, ίδια λογική με το Logistic Regression (balanced)
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
class_weight = {0: weights[0], 1: weights[1]}
print("Βάρη κλάσεων:", class_weight)

# ---- Μοντέλο ----
mlp_simple = models.Sequential([
    layers.Input(shape=(Xtr.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
mlp_simple.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
mlp_simple.summary()

# ---- Εκπαίδευση ----
history = mlp_simple.fit(Xtr, ytr,
                         validation_data=(Xva, yva),
                         epochs=50, batch_size=32,
                         class_weight=class_weight, verbose=0)

# ---- Καμπύλες εκπαίδευσης ----
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Απλό MLP: καμπύλες loss")
plt.savefig("figures/fig6_mlp_simple_loss.png", bbox_inches="tight"); plt.show()

# ---- Αξιολόγηση στο validation ----
def evaluate_keras(model, X, y, threshold=0.5):
    """Οι ίδιες πέντε μετρικές με το baseline."""
    proba = model.predict(X, verbose=0).ravel()
    pred = (proba >= threshold).astype(int)
    return {"Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred),
            "F1-score": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, proba)}

print(pd.Series(evaluate_keras(mlp_simple, Xva, yva)).round(3))

# ============================================================
# 3.2 ΠΡΟΗΓΜΕΝΕΣ ΤΕΧΝΙΚΕΣ
# ============================================================

n_features = Xtr.shape[1]
results_val = {}   # αποθήκευση μετρικών validation για τον πίνακα σύγκρισης
results_val["LR (balanced)"] = evaluate(lr_bal, X_val, y_val)
results_val["MLP (απλό)"] = evaluate_keras(mlp_simple, Xva, yva)

def train_and_eval(model, name, epochs=50):
    """Εκπαίδευση, καμπύλες loss και μετρικές validation για κάθε παραλλαγή."""
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                     batch_size=32, class_weight=class_weight, verbose=0)
    results_val[name] = evaluate_keras(model, Xva, yva)
    print(name, pd.Series(results_val[name]).round(3).to_dict())
    return hist

# ---- Παραλλαγή Α: BatchNorm + Dropout + L2 ----
tf.keras.utils.set_random_seed(SEED)
inp = layers.Input(shape=(n_features,))
x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(inp)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation="sigmoid")(x)
mlp_reg = models.Model(inp, out)
mlp_reg.summary()
hist_reg = train_and_eval(mlp_reg, "MLP + BN + Dropout + L2")

# ---- Παραλλαγή Β: Residual (skip) connections ----
def res_block(x, units, dropout=0.3, l2=1e-4):
    """Residual block: x + F(x), όπου F = Dense-BN-ReLU-Dropout-Dense-BN."""
    shortcut = x
    h = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))(h)
    h = layers.BatchNormalization()(h)
    h = layers.Add()([shortcut, h])          # skip connection
    return layers.Activation("relu")(h)

tf.keras.utils.set_random_seed(SEED)
inp = layers.Input(shape=(n_features,))
x = layers.Dense(64, activation="relu")(inp)   # προβολή στο ίδιο πλάτος (64)
x = res_block(x, 64)
x = res_block(x, 64)
out = layers.Dense(1, activation="sigmoid")(x)
mlp_res = models.Model(inp, out)
mlp_res.summary()
hist_res = train_and_eval(mlp_res, "MLP + Residual")

# ---- Πίνακας σύγκρισης (validation) ----
print(pd.DataFrame(results_val).T.round(3))

# ---- Καμπύλες val loss για όλες τις παραλλαγές ----
plt.figure(figsize=(8, 4))
for name, h in [("Απλό MLP", history), ("BN+Dropout+L2", hist_reg), ("Residual", hist_res)]:
    plt.plot(h.history["val_loss"], label=name)
plt.xlabel("Epoch"); plt.ylabel("Validation loss"); plt.legend()
plt.title("Σύγκριση των παραλλαγών (val loss)")
plt.savefig("figures/fig7_variants_val_loss.png", bbox_inches="tight"); plt.show()

# ============================================================
# 3.3 ΣΥΝΤΟΝΙΣΜΟΣ ΥΠΕΡΠΑΡΑΜΕΤΡΩΝ (KerasTuner)
# ============================================================

# ---- Early stopping (σταματά όταν το val loss δεν βελτιώνεται) ----
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                     restore_best_weights=True)

# ---- Συνάρτηση που χτίζει το μοντέλο για κάθε συνδυασμό ----
def build_model(hp):
    # Είδος και ένταση κανονικοποίησης (L1, L2 ή και τα δύο)
    reg_type = hp.Choice("reg_type", ["l1", "l2", "l1_l2"])
    reg_val = hp.Float("reg_strength", 1e-5, 1e-2, sampling="log")
    if reg_type == "l1":
        reg = regularizers.l1(reg_val)
    elif reg_type == "l2":
        reg = regularizers.l2(reg_val)
    else:
        reg = regularizers.l1_l2(l1=reg_val, l2=reg_val)

    use_bn = hp.Boolean("batch_norm")
    dropout = hp.Float("dropout", 0.0, 0.5, step=0.1)

    inp = layers.Input(shape=(n_features,))
    x = inp
    for i in range(hp.Int("n_layers", 1, 3)):
        x = layers.Dense(hp.Int(f"units_{i}", 16, 128, step=16),
                         kernel_regularizer=reg)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

# ---- Αναζήτηση ----
tf.keras.utils.set_random_seed(SEED)
tuner = kt.RandomSearch(build_model,
                        objective=kt.Objective("val_auc", direction="max"),
                        max_trials=20, seed=SEED,
                        directory="tuner_dir", project_name="churn_mlp",
                        overwrite=True)
tuner.search(Xtr, ytr, validation_data=(Xva, yva),
             epochs=100, batch_size=32,
             class_weight=class_weight, callbacks=[early_stop], verbose=0)

# ---- Καλύτεροι συνδυασμοί ----
best_hp = tuner.get_best_hyperparameters(1)[0]
print("Καλύτερες υπερπαράμετροι:", best_hp.values)

# ---- Επανεκπαίδευση του καλύτερου μοντέλου ----
tf.keras.utils.set_random_seed(SEED)
mlp_tuned = build_model(best_hp)
mlp_tuned.summary()
hist_tuned = mlp_tuned.fit(Xtr, ytr, validation_data=(Xva, yva),
                           epochs=100, batch_size=32,
                           class_weight=class_weight, callbacks=[early_stop], verbose=0)
print("Σταμάτησε στην epoch:", len(hist_tuned.history["loss"]))

# ---- Αξιολόγηση στο validation (το test παραμένει ανέγγιχτο) ----
results_val["MLP tuned"] = evaluate_keras(mlp_tuned, Xva, yva)
print(pd.DataFrame(results_val).T.round(3))

# ---- Καμπύλες loss ----
plt.figure(figsize=(8, 4))
plt.plot(hist_tuned.history["loss"], label="Train loss")
plt.plot(hist_tuned.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Βελτιστοποιημένο MLP: καμπύλες loss")
plt.savefig("figures/fig9_tuned_loss.png", bbox_inches="tight"); plt.show()

# ============================================================
# 3.4 ΕΛΕΓΧΟΣ OVERFITTING: ΣΥΓΚΡΙΣΗ ΜΕΘΟΔΩΝ
# ============================================================

def build_variant(dropout, l2):
    """Ίδια αρχιτεκτονική με το tuned μοντέλο, με ρυθμιζόμενο dropout και L2."""
    reg = regularizers.l2(l2) if l2 > 0 else None
    inp = layers.Input(shape=(n_features,))
    x = inp
    for u in [64, 64, 16]:
        x = layers.Dense(u, activation="relu", kernel_regularizer=reg)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(0.0009),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m

experiments = {
    "Χωρίς καμία μέθοδο": dict(dropout=0.0, l2=0.0, stop=False),
    "Μόνο dropout": dict(dropout=0.2, l2=0.0, stop=False),
    "Dropout + L2": dict(dropout=0.2, l2=4.7e-5, stop=False),
    "Dropout + L2 + early stopping": dict(dropout=0.2, l2=4.7e-5, stop=True),
}

plt.figure(figsize=(9, 4))
for name, cfg in experiments.items():
    tf.keras.utils.set_random_seed(SEED)
    m = build_variant(cfg["dropout"], cfg["l2"])
    cb = [early_stop] if cfg["stop"] else []
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=100, batch_size=32,
              class_weight=class_weight, callbacks=cb, verbose=0)
    plt.plot(h.history["val_loss"], label=name)
    print(name, "| ελάχιστο val loss:", round(min(h.history["val_loss"]), 3),
          "| τελικό val loss:", round(h.history["val_loss"][-1], 3),
          "| epochs:", len(h.history["val_loss"]))
plt.xlabel("Epoch"); plt.ylabel("Validation loss"); plt.legend()
plt.title("Επίδραση των μεθόδων ελέγχου του overfitting")
plt.savefig("figures/fig10_overfitting_methods.png", bbox_inches="tight"); plt.close()

# ============================================================
# 3.5 ΑΞΙΟΛΟΓΗΣΗ ΣΤΟ TEST SET ΚΑΙ ΣΥΓΚΡΙΣΗ ΜΕ ΤΟ BASELINE
# ============================================================

# 1. Αξιολόγηση του τελικού baseline (Logistic Regression - balanced) στο test set
lr_test_metrics = evaluate(lr_bal, X_test, y_test)

# 2. Αξιολόγηση του βελτιστοποιημένου νευρωνικού δικτύου (MLP tuned) στο test set
mlp_test_metrics = evaluate_keras(mlp_tuned, Xte, yte)

# 3. Συγκεντρωτικός πίνακας σύγκρισης στο test set
comparison_test = pd.DataFrame({
    "Logistic Regression (balanced)": lr_test_metrics,
    "MLP Tuned (TensorFlow/Keras)": mlp_test_metrics
}).T

comparison_test["Diff (MLP - LR)"] = comparison_test.loc["MLP Tuned (TensorFlow/Keras)"] - comparison_test.loc["Logistic Regression (balanced)"]
print("=== Σύγκριση μοντέλων στο Test Set ===")
print(comparison_test.round(3))

# 4. Πιθανότητες και προβλέψεις στο test set (threshold = 0.5)
threshold = 0.5
lr_proba_test = lr_bal.predict_proba(X_test)[:, 1]
lr_pred_test = (lr_proba_test >= threshold).astype(int)

mlp_proba_test = mlp_tuned.predict(Xte, verbose=0).ravel()
mlp_pred_test = (mlp_proba_test >= threshold).astype(int)

print("\nConfusion Matrix - Logistic Regression (Test):")
print(confusion_matrix(y_test, lr_pred_test))

print("\nConfusion Matrix - MLP Tuned (Test):")
print(confusion_matrix(yte, mlp_pred_test))

# 5. Οπτική σύγκριση: Confusion Matrices δίπλα-δίπλα
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ConfusionMatrixDisplay.from_predictions(
    y_test, lr_pred_test,
    display_labels=["Stayed", "Churned"],
    cmap="Blues", ax=axes[0], colorbar=False
)
axes[0].set_title("Logistic Regression (Test)")

ConfusionMatrixDisplay.from_predictions(
    yte, mlp_pred_test,
    display_labels=["Stayed", "Churned"],
    cmap="Oranges", ax=axes[1], colorbar=False
)
axes[1].set_title("MLP Tuned (Test)")

plt.tight_layout()
plt.savefig("figures/fig11_test_confusion_comparison.png", bbox_inches="tight")
plt.show()

# 6. Οπτική σύγκριση: Καμπύλες ROC στο ίδιο διάγραμμα
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba_test)
auc_lr = roc_auc_score(y_test, lr_proba_test)

fpr_mlp, tpr_mlp, _ = roc_curve(yte, mlp_proba_test)
auc_mlp = roc_auc_score(yte, mlp_proba_test)

plt.figure(figsize=(7, 5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_mlp, tpr_mlp, label=f"MLP Tuned (AUC = {auc_mlp:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (AUC = 0.500)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Σύγκριση καμπυλών ROC στο Test Set")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("figures/fig12_test_roc_comparison.png", bbox_inches="tight")
plt.show()

# ============================================================
# 3.6 ΕΡΜΗΝΕΥΣΙΜΟΤΗΤΑ ΜΕ SHAP (MLP TUNED)
# ============================================================

# Δείγμα αναφοράς (background) από το train set και δείγμα ελέγχου από το test set
rng = np.random.default_rng(SEED)
bg_idx = rng.choice(len(Xtr), size=min(200, len(Xtr)), replace=False)
test_idx = rng.choice(len(Xte), size=min(300, len(Xte)), replace=False)

X_bg = Xtr[bg_idx]
X_explain = Xte[test_idx]

# Υπολογισμός SHAP values με GradientExplainer για το μοντέλο Keras
explainer = shap.GradientExplainer(mlp_tuned, X_bg)
shap_values = explainer.shap_values(X_explain)

# Προσαρμογή σχήματος σε περίπτωση που επιστρέφει λίστα ή επιπλέον διάσταση
if isinstance(shap_values, list):
    sv = shap_values[0]
else:
    sv = shap_values
sv = np.squeeze(sv)

# Κατάταξη χαρακτηριστικών με βάση το μέσο απόλυτο SHAP value
mean_abs_shap = pd.Series(
    np.abs(sv).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

print("Top 10 χαρακτηριστικά κατά μέσο |SHAP|:")
print(mean_abs_shap.head(10).round(4))

# Summary plot (beeswarm)
plt.figure(figsize=(8, 5))
shap.summary_plot(
    sv,
    pd.DataFrame(X_explain, columns=X_test.columns),
    show=False,
    max_display=10
)
plt.title("SHAP Summary Plot — MLP Tuned")
plt.savefig("figures/fig13_shap_summary.png", bbox_inches="tight")
plt.close()

# Bar plot μέσης σημασίας χαρακτηριστικών
plt.figure(figsize=(8, 4.5))
shap.summary_plot(
    sv,
    pd.DataFrame(X_explain, columns=X_test.columns),
    plot_type="bar",
    show=False,
    max_display=10
)
plt.title("Μέση Σημασία Χαρακτηριστικών (Mean |SHAP|) — MLP Tuned")
plt.savefig("figures/fig14_shap_bar.png", bbox_inches="tight")
plt.close()

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# ============================================================
# 4.2 ΣΥΓΚΡΙΤΙΚΗ ΑΞΙΟΛΟΓΗΣΗ (RANDOM FOREST & XGBOOST)
# ============================================================

def eval_probs(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1-score": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, proba)
    }

# 1. Random Forest με class_weight="balanced"
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_metrics = eval_probs(y_test, rf.predict_proba(X_test)[:, 1])

# 2. XGBoost (με fallback σε HistGradientBoosting αν δεν είναι εγκατεστημένο το xgboost)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=SEED
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_label = "XGBoost (weighted)"
except ImportError:
    xgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=SEED
    )
    sample_w = np.where(y_train == 1, scale_pos_weight, 1.0)
    xgb.fit(X_train, y_train, sample_weight=sample_w)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_label = "Gradient Boosting (weighted)"

xgb_metrics = eval_probs(y_test, xgb_proba)

# Συγκεντρωτικός πίνακας σύγκρισης όλων των μοντέλων στο test set
benchmark_table = pd.DataFrame({
    "Logistic Regression (balanced)": evaluate(lr_bal, X_test, y_test),
    "MLP Tuned": evaluate_keras(mlp_tuned, Xte, yte),
    "Random Forest (balanced)": rf_metrics,
    xgb_label: xgb_metrics
}).T

print(benchmark_table.round(3))