import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    balanced_accuracy_score, 
    precision_score, 
    recall_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SEED = 42
ROWS_PER_DAY = 96

# --- CORE UTILITIES ---

def clean_and_engineer(df):
    df = df.drop_duplicates().dropna()
    df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df = pd.get_dummies(df, columns=['Day of the week'], drop_first=False)
    return df

def get_features(df):
    day_cols = [c for c in df.columns if 'Day of the week_' in c]
    return ['Hour_Sin', 'Hour_Cos', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount'] + day_cols

def time_split(X, y, test_size=0.2):
    n_days = len(X) // ROWS_PER_DAY
    max_rows = n_days * ROWS_PER_DAY
    X, y = X.iloc[:max_rows], y[:max_rows]
    split_day = n_days - max(1, int(round(n_days * test_size)))
    train_idx = np.arange(0, split_day * ROWS_PER_DAY)
    test_idx = np.arange(split_day * ROWS_PER_DAY, n_days * ROWS_PER_DAY)
    return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]

def inject_noise(X, seed=SEED):
    """Simulates occlusion (±1-3 count error) and 2% sensor dropout."""
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    counts = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']
    for col in counts:
        Xn[col] = (Xn[col] + rng.integers(-3, 4, size=len(Xn))).clip(lower=0)
    drop_mask = rng.random(len(Xn)) < 0.02
    for r in np.flatnonzero(drop_mask):
        Xn.iat[r, Xn.columns.get_loc(rng.choice(counts))] = 0
    return Xn

def plot_cm(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# --- MODEL ARCHITECTURE ---

def build_nn(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- MAIN EXPERIMENT ---

def main():
    out_dir = Path("outputs/traffic_final_research")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = clean_and_engineer(pd.read_csv("TrafficTwoMonth.csv"))
    le = LabelEncoder()
    y = le.fit_transform(df["Traffic Situation"])
    X = df[get_features(df)]
    X_train, X_test, y_train, y_test = time_split(X, y)
    
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    X_test_noisy_s = scaler.transform(inject_noise(X_test))

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=600, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=6, random_state=SEED),
        "NeuralNetwork": build_nn(X_train_s.shape[1], len(le.classes_))
    }

    results = []
    
    for name, model in models.items():
        print(f"Processing: {name}")
        if name == "NeuralNetwork":
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train_s, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.1, callbacks=[es])
            c_preds = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
            n_preds = np.argmax(model.predict(X_test_noisy_s, verbose=0), axis=1)
            model.save(out_dir / "traffic_nn_model.keras")
        else:
            model.fit(X_train_s, y_train)
            c_preds = model.predict(X_test_s)
            n_preds = model.predict(X_test_noisy_s)

        for setting, preds in [("Clean", c_preds), ("Noisy", n_preds)]:
            results.append({
                "Model": name,
                "Setting": setting,
                "Accuracy": accuracy_score(y_test, preds),
                "BalAcc": balanced_accuracy_score(y_test, preds),
                "F1_Macro": f1_score(y_test, preds, average="macro"),
                "Precision_Macro": precision_score(y_test, preds, average="macro", zero_division=0),
                "Recall_Macro": recall_score(y_test, preds, average="macro")
            })
            plot_cm(y_test, preds, le.classes_, f"{name} ({setting})", out_dir / f"cm_{name.lower()}_{setting.lower()}.png")

    res_df = pd.DataFrame(results)
    
    # Generate Decay Plots
    pivot = res_df.pivot(index="Model", columns="Setting", values="F1_Macro")
    pivot['Decay'] = pivot['Clean'] - pivot['Noisy']
    pivot = pivot.sort_values(by='Decay')
    
    plt.figure(figsize=(10, 6))
    pivot['Decay'].plot(kind='barh', color='#e74c3c')
    plt.title("Robustness Decay: F1-Macro Loss Under Sensor Noise", fontsize=14)
    plt.xlabel("Performance Drop (Lower is Better)")
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_decay.png")

    # Final Output
    print("\n" + "="*90)
    print(" COMPREHENSIVE TRAFFIC CLASSIFICATION RESULTS")
    print("="*90)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(res_df.sort_values(by=["Setting", "F1_Macro"], ascending=[True, False]))
    
    res_df.to_csv(out_dir / "full_experiment_results.csv", index=False)
    joblib.dump(scaler, out_dir / "traffic_scaler.pkl")
    joblib.dump(le, out_dir / "traffic_encoder.pkl")
    print(f"\nArtifacts saved to: {out_dir}")

if __name__ == "__main__":
    main()
