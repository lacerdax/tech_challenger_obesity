import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # limpeza básica para evitar "bugs de espaço"
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def build_preprocess(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    return preprocess, num_cols, cat_cols


def evaluate(model, X_test, y_test, title: str) -> float:
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"\n=== {title} ===")
    print("Accuracy:", round(acc, 4))
    print("\nClassification report:\n", classification_report(y_test, pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    return acc


def main():
    # Dataset
    csv_path = os.path.join("data", "Obesity.csv")
    target = "Obesity"  # ✅ alvo real do seu CSV

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path}\n"
            "Coloque o Obesity.csv dentro da pasta data/ com esse nome exato."
        )

    df = load_data(csv_path)

    if target not in df.columns:
        raise ValueError(
            f"Coluna alvo '{target}' não encontrada.\n"
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    preprocess, num_cols, cat_cols = build_preprocess(X)

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("✅ Dataset carregado:", df.shape)
    print("✅ Coluna alvo:", target)
    print("✅ Numéricas:", num_cols)
    print("✅ Categóricas:", cat_cols)
    print("\nDistribuição do alvo:\n", y.value_counts())

    # 1) Baseline: Logistic Regression (multiclasse)
    logreg = LogisticRegression(
    max_iter=5000,
    n_jobs=-1
    )

    logreg_pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", logreg),
    ])

    logreg_pipe.fit(X_train, y_train)
    acc_lr = evaluate(logreg_pipe, X_test, y_test, "LogisticRegression (baseline)")

    # 2) Modelo forte: RandomForest + tuning leve
    rf = RandomForestClassifier(
        n_estimators=700,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", rf),
    ])

    param_grid = {
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_leaf": [1, 2, 4],
        "model__min_samples_split": [2, 5, 10],
    }

    grid = GridSearchCV(
        rf_pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("\nBest params (RandomForest):", grid.best_params_)
    acc_rf = evaluate(best_model, X_test, y_test, "RandomForest (tuned)")

    # Escolhe o melhor
    if acc_rf >= acc_lr:
        final_model = best_model
        final_acc = acc_rf
        final_name = "RandomForest (tuned)"
        final_params = grid.best_params_
    else:
        final_model = logreg_pipe
        final_acc = acc_lr
        final_name = "LogisticRegression (baseline)"
        final_params = {}

    # Salvar
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "obesity_model.joblib")
    joblib.dump(final_model, model_path)

    metrics = {
        "final_model": final_name,
        "accuracy": float(final_acc),
        "best_params_if_rf": final_params,
        "random_state": RANDOM_STATE,
        "target": target,
        "num_features": num_cols,
        "cat_features": cat_cols,
        "dataset_shape": [int(df.shape[0]), int(df.shape[1])]
    }

    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n✅ Modelo final:", final_name)
    print("✅ Accuracy final:", round(final_acc, 4))
    print("✅ Salvo em:", model_path)
    print("✅ Métricas salvas em:", metrics_path)

    # Checagem do requisito
    if final_acc < 0.75:
        print("\n⚠️ ALERTA: accuracy < 0.75 (abaixo da meta do desafio).")


if __name__ == "__main__":
    main()
