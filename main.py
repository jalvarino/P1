# main.py
import os
import time
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para guardar imágenes sin GUI
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore")


# ---------------------------
# Utilidades
# ---------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró el archivo: {csv_path}\n"
            f"Colócalo en ./data/ai4i2020.csv o indica la ruta correcta con --csv."
        )
    df = pd.read_csv(csv_path)

    # Target puede llamarse 'Machine failure' o 'Target' (o variantes)
    candidates = ['Machine failure', 'Target', 'machine_failure', 'Failure']
    target_col = next((c for c in candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(
            "No se encontró la columna objetivo. Asegúrate de que exista "
            "'Machine failure' o 'Target' en el CSV."
        )
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Remover identificadores si existen
    drop_cols = [c for c in ['UDI', 'Product ID', 'product_id', 'id'] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors='ignore')

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )
    return pre, num_cols, cat_cols


# ---------------------------
# Implementación KDE + NB
# ---------------------------
class KDEPDF:
    """
    Estimador de densidad univariante por KDE.

    Modos:
    - 'silverman': kernel gaussiano con bandwidth de la regla de Silverman (posible escala h_scale)
    - 'gaussian_opt': kernel gaussiano con bandwidth optimizado por verosimilitud LOO (referencia)
    - 'parzen': ventana simple (tophat|triangular) con bandwidth por defecto tipo Silverman*escala o fijo.
    """
    def __init__(self, mode='silverman', bandwidth=None, kernel='tophat',
                 grid=None, h_scale=1.0, random_state=RANDOM_STATE):
        self.mode = mode
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.grid = grid
        self.h_scale = h_scale
        self.random_state = random_state
        self.eps = 1e-12

    def fit(self, x: np.ndarray):
        x = np.asarray(x).reshape(-1)
        self.n_ = len(x)
        self.std_ = np.std(x, ddof=1) if self.n_ > 1 else 0.0

        if self.mode == 'silverman':
            base_h = 1.06 * (self.std_ + self.eps) * (max(self.n_, 2) ** (-1/5))
            self.h_ = max(base_h * self.h_scale, self.eps)

        elif self.mode == 'gaussian_opt':
            # Optimización por verosimilitud LOO sobre una rejilla logarítmica relativa a std
            if self.grid is None:
                base = max(self.std_, 1e-3)
                self.grid = np.logspace(-2, 1, 15) * base
            best_ll = -np.inf
            best_h = None
            rng = np.random.RandomState(self.random_state)

            idx = np.arange(self.n_)
            if self.n_ > 200:
                idx = rng.choice(idx, size=200, replace=False)
            xs = x[idx]
            for h in self.grid:
                ll = 0.0
                for i in range(len(xs)):
                    xi = xs[i]
                    diff = xi - np.delete(xs, i)
                    val = np.exp(-0.5 * (diff / h) ** 2) / (math.sqrt(2 * math.pi) * h)
                    dens = np.maximum(val.mean(), self.eps)
                    ll += np.log(dens)
                if ll > best_ll:
                    best_ll, best_h = ll, h
            self.h_ = float(best_h)

        elif self.mode == 'parzen':
            if self.bandwidth is None:
                base_h = 1.06 * (self.std_ + self.eps) * (max(self.n_, 2) ** (-1/5))
                self.h_ = max(base_h * self.h_scale, self.eps)
            else:
                self.h_ = float(self.bandwidth)
        else:
            raise ValueError("Modo desconocido para KDEPDF")

        self.x_ = x
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1)
        h = max(self.h_, self.eps)

        if self.mode in ('silverman', 'gaussian_opt'):
            diff = (X[:, None] - self.x_[None, :]) / h
            vals = np.exp(-0.5 * diff**2) / (math.sqrt(2 * math.pi) * h)
            dens = np.maximum(vals.mean(axis=1), self.eps)
            return np.log(dens)

        elif self.mode == 'parzen':
            u = (X[:, None] - self.x_[None, :]) / h
            if self.kernel == 'tophat':
                K = (np.abs(u) <= 1).astype(float)
            elif self.kernel == 'triangular':
                K = np.clip(1 - np.abs(u), 0, 1)
            else:
                raise ValueError("Kernel Parzen no soportado. Use 'tophat' o 'triangular'.")
            dens = np.maximum(K.mean(axis=1) / h, self.eps)
            return np.log(dens)

        else:
            raise ValueError("Modo desconocido para KDEPDF")


class NaiveBayesKDE(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes con verosimilitudes univariantes estimadas por KDE.
    """
    def __init__(self, mode='silverman', parzen_kernel='tophat', parzen_bandwidth=None,
                 h_scale=1.0, balanced_priors=False, random_state=RANDOM_STATE):
        self.mode = mode
        self.parzen_kernel = parzen_kernel
        self.parzen_bandwidth = parzen_bandwidth
        self.h_scale = h_scale
        self.balanced_priors = balanced_priors
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.n_features_ = X.shape[1]

        # Priors
        if self.balanced_priors:
            self.class_log_prior_ = {c: math.log(1.0 / len(self.classes_)) for c in self.classes_}
        else:
            priors = {c: cnt / len(y) for c, cnt in zip(self.classes_, counts)}
            self.class_log_prior_ = {c: math.log(p) for c, p in priors.items()}

        # Ajustar KDE por clase y por feature
        self.kdes_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.kdes_[c] = []
            for j in range(self.n_features_):
                pdf = KDEPDF(mode=self.mode,
                             bandwidth=self.parzen_bandwidth,
                             kernel=self.parzen_kernel,
                             h_scale=self.h_scale,
                             random_state=self.random_state)
                pdf.fit(Xc[:, j])
                self.kdes_[c].append(pdf)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            ll = np.zeros(X.shape[0])
            for j in range(self.n_features_):
                ll += self.kdes_[c][j].score_samples(X[:, j])
            jll[:, idx] = ll + self.class_log_prior_[c]
        return jll

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        a = jll - jll.max(axis=1, keepdims=True)
        log_prob = a - np.log(np.exp(a).sum(axis=1, keepdims=True))
        return log_prob

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        idx = jll.argmax(axis=1)
        return self.classes_[idx]


# ---------------------------
# Evaluación y visualizaciones
# ---------------------------
def evaluate_models(
    X_df: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    parzen_kernel: str = "tophat",
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    CV externa 5-fold. Para KDE Gauss (Silverman) se hace tuning de h_scale con CV interna 3-fold.
    """
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    h_grid = [0.5, 0.75, 1.0, 1.25, 1.5]  # escala sobre Silverman
    models_fixed = {
        "GaussianNB": GaussianNB(),
        "KDE_Parzen": NaiveBayesKDE(mode='parzen', parzen_kernel=parzen_kernel, balanced_priors=False, random_state=random_state),
        "KDE_Gaussian_LOO": NaiveBayesKDE(mode='gaussian_opt', balanced_priors=False, random_state=random_state),
    }
    tuned_name = "KDE_Gaussian_Silverman_CV"  # con busca de h_scale vía inner-CV
    results = []
    roc_curves = {}

    print("\n=== Evaluación 5-fold (AUC-ROC) ===")
    for name, base_model in list(models_fixed.items()) + [(tuned_name, None)]:
        aucs, fit_times, pred_times = [], [], []
        y_true_all, y_score_all = [], []
        chosen_hs = []  # para Silverman_CV

        for fold_idx, (tr, te) in enumerate(outer.split(X_df, y), start=1):
            X_tr, X_te = X_df.iloc[tr], X_df.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]

            if name == tuned_name:
                # Inner CV para seleccionar h_scale que maximiza AUC
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
                best_auc, best_h = -np.inf, None
                for hs in h_grid:
                    auc_inner = []
                    for itr, ite in inner.split(X_tr, y_tr):
                        Xi_tr, Xi_te = X_tr.iloc[itr], X_tr.iloc[ite]
                        yi_tr, yi_te = y_tr.iloc[itr], y_tr.iloc[ite]
                        pipe_i = Pipeline([
                            ("pre", preprocessor),
                            ("clf", NaiveBayesKDE(mode='silverman', h_scale=hs, balanced_priors=False, random_state=random_state))
                        ])
                        pipe_i.fit(Xi_tr, yi_tr)
                        proba_i = pipe_i.predict_proba(Xi_te)[:, 1]
                        auc_inner.append(roc_auc_score(yi_te, proba_i))
                    mean_auc = float(np.mean(auc_inner))
                    if mean_auc > best_auc:
                        best_auc, best_h = mean_auc, hs
                model = NaiveBayesKDE(mode='silverman', h_scale=best_h, balanced_priors=False, random_state=random_state)
                chosen_hs.append(best_h)
            else:
                model = deepcopy(base_model)

            pipe = Pipeline([("pre", preprocessor), ("clf", model)])

            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            t1 = time.perf_counter()
            fit_times.append(t1 - t0)

            t0 = time.perf_counter()
            proba = pipe.predict_proba(X_te)[:, 1]
            t1 = time.perf_counter()
            pred_times.append(t1 - t0)

            aucs.append(roc_auc_score(y_te, proba))
            y_true_all.append(y_te.values)
            y_score_all.append(proba)

        y_true_all = np.concatenate(y_true_all)
        y_score_all = np.concatenate(y_score_all)
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        roc_curves[name] = (fpr, tpr)

        row = {
            "Modelo": name,
            "AUC_media_5fold": float(np.mean(aucs)),
            "AUC_std": float(np.std(aucs)),
            "Tiempo_entrenamiento_prom(s)": float(np.mean(fit_times)),
            "Tiempo_prediccion_prom(s)": float(np.mean(pred_times)),
        }
        results.append(row)

        # Resumen consola
        print(f"- {name:>28s} | AUC: {row['AUC_media_5fold']:.4f} ± {row['AUC_std']:.4f} | "
              f"Train(s): {row['Tiempo_entrenamiento_prom(s)']:.4f} | Pred(s): {row['Tiempo_prediccion_prom(s)']:.4f}")
        if name == tuned_name:
            print(f"    h_scale seleccionados por fold: {chosen_hs}")

    results_df = pd.DataFrame(results).sort_values(by="AUC_media_5fold", ascending=False)
    return results_df, roc_curves


def plot_roc(roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]], outdir: str):
    ensure_dir(os.path.join(outdir, "img"))
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('FPR (1 - Especificidad)')
    plt.ylabel('TPR (Sensibilidad)')
    plt.title('Curvas ROC (validación cruzada consolidada)')
    plt.legend()
    plt.grid(True, alpha=0.25)
    out_path = os.path.join(outdir, "img", "roc_comparadas.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] ROC guardada en: {out_path}")


def plot_densities_by_class(X_raw: pd.DataFrame, y: pd.Series,
                            feature: str, outdir: str):
    """
    Grafica densidades KDE por clase para una variable del espacio original (sin escalar).
    """
    ensure_dir(os.path.join(outdir, "img"))
    if feature not in X_raw.columns:
        print(f"[WARN] La variable '{feature}' no existe en X. Se omite densidad.")
        return
    if not pd.api.types.is_numeric_dtype(X_raw[feature]):
        print(f"[WARN] La variable '{feature}' no es numérica. Se omite densidad.")
        return

    x0 = X_raw[y == 0][feature].values
    x1 = X_raw[y == 1][feature].values

    xs = np.linspace(np.percentile(X_raw[feature], 1),
                     np.percentile(X_raw[feature], 99), 300)

    setups = [
        ("silverman", "Silverman"),
        ("gaussian_opt", "Gauss LOO"),
        ("parzen", "Parzen (tophat)")
    ]
    for mode, label in setups:
        kde0 = KDEPDF(mode=mode, kernel='tophat').fit(x0)
        kde1 = KDEPDF(mode=mode, kernel='tophat').fit(x1)
        d0 = np.exp(kde0.score_samples(xs))
        d1 = np.exp(kde1.score_samples(xs))

        plt.figure(figsize=(8, 4.5))
        plt.plot(xs, d0, label='Clase 0')
        plt.plot(xs, d1, label='Clase 1')
        plt.title(f"Densidades por clase - {feature} - {label}")
        plt.xlabel(feature)
        plt.ylabel("Densidad")
        plt.legend()
        plt.grid(True, alpha=0.25)
        out_path = os.path.join(outdir, "img", f"densidad_{feature}_{mode}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[OK] Densidad guardada en: {out_path}")


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Proyecto 1: Naive Bayes con KDE para Mantenimiento Predictivo (AI4I 2020)"
    )
    p.add_argument("--csv", required=True, help="Ruta del ai4i2020.csv (ej: ./data/ai4i2020.csv)")
    p.add_argument("--outdir", default="./outputs", help="Carpeta de salida (resultados y figuras)")
    p.add_argument("--parzen-kernel", default="tophat", choices=["tophat", "triangular"],
                   help="Kernel para KDE tipo Parzen (default: tophat)")
    p.add_argument("--seed", type=int, default=RANDOM_STATE, help="Semilla aleatoria")
    p.add_argument("--densidad-feature", default=None,
                   help="Nombre de variable numérica para densidades (si se omite, se toma la primera numérica disponible)")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "img"))

    # 1) Cargar dataset
    X_raw, y = load_dataset(args.csv)
    print(f"Forma X: {X_raw.shape} | y positivos: {y.sum()}/{len(y)} = {y.mean():.4f}")

    # 2) Preprocesamiento
    pre, num_cols, _ = build_preprocessor(X_raw)

    # 3) Evaluación y comparación de modelos
    results_df, roc_curves = evaluate_models(
        X_df=X_raw,
        y=y,
        preprocessor=pre,
        parzen_kernel=args.parzen_kernel,
        random_state=args.seed
    )

    # 4) Guardar resultados (CSV) y ROC
    out_csv = os.path.join(args.outdir, "resultados_auc.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[OK] Resultados guardados en: {out_csv}")
    print("\nTabla resumen:")
    print(results_df.to_string(index=False))

    plot_roc(roc_curves, args.outdir)

    # 5) Densidades por clase (opcional pero recomendado por la guía)
    feat = args.densidad_feature
    if feat is None:
        # Tomar la primera numérica disponible del espacio original
        num_candidates = [c for c in num_cols if pd.api.types.is_numeric_dtype(X_raw[c])]
        if len(num_candidates) > 0:
            feat = num_candidates[0]
    if feat is not None:
        plot_densities_by_class(X_raw, y, feat, args.outdir)
    else:
        print("[INFO] No se encontraron variables numéricas para densidades.")

    print("\n[LISTO] Ejecución finalizada. Revisa la carpeta de salidas.")


if __name__ == "__main__":
    main()