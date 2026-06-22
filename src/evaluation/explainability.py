import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from src.models.base import get_xgb_params, DISEASES


def train_final_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train on full dataset (no CV) for SHAP/imodels analysis."""
    model = XGBClassifier(**get_xgb_params())
    model.fit(X, y, verbose=False)
    return model


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def plot_shap_global(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str = "shap_global.png",
) -> None:
    """Mean |SHAP| across all classes — shows overall feature importance."""
    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    mean_importance = pd.Series(
        mean_abs.mean(axis=0), index=X.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    mean_importance.head(21).plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance — Model C")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str = "shap_beeswarm.png",
    class_idx: int = 0,
) -> None:
    """SHAP beeswarm for one disease class."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[class_idx],
        X,
        plot_type="dot",
        show=False,
        max_display=21,
    )
    plt.title(f"SHAP Feature Importance — {DISEASES[class_idx]}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def extract_imodels_rules(
    X: pd.DataFrame,
    y: pd.Series,
    max_rules: int = 15,
) -> list[str]:
    """
    Fits RuleFit on Model C features. Returns post-hoc IF-THEN rules.
    Note: distinct from hand-crafted YAML rules in the symbolic engine.
    """
    try:
        from imodels import RuleFitClassifier
        rulefit = RuleFitClassifier(max_rules=max_rules, random_state=42)
        rulefit.fit(X.values, y.values, feature_names=list(X.columns))
        rules = rulefit.get_rules()
        rules = rules[rules["coef"] != 0].sort_values("importance", ascending=False)
        return [
            f"IF {row['rule']} THEN importance={row['importance']:.4f}"
            for _, row in rules.head(max_rules).iterrows()
        ]
    except Exception as e:
        return [f"imodels rule extraction failed: {e}"]
