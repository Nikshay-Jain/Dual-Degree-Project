import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# ==================== COST FUNCTIONS (SAME AS BEFORE) ====================
def calculate_costs(sum_insured, y_true, y_pred):
    """FN: 90% of sum_insured, FP: £100"""
    sum_insured = np.asarray(sum_insured)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    costs = np.zeros(len(y_true), dtype=float)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    costs[fn_mask] = sum_insured[fn_mask] * 0.90
    costs[fp_mask] = 100.0
    return costs

def evaluate_cost_sensitive(y_true, y_pred, sum_insured):
    """Comprehensive cost-sensitive evaluation"""
    costs = calculate_costs(sum_insured, y_true, y_pred)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        rocauc = roc_auc_score(y_true, y_pred)
    except:
        rocauc = np.nan

    return {
        'total_cost': float(costs.sum()),
        'fn_cost': float(costs[fn_mask].sum()),
        'fp_cost': float(costs[fp_mask].sum()),
        'fn_count': int(fn_mask.sum()),
        'fp_count': int(fp_mask.sum()),
        'tp': int(((y_true == 1) & (y_pred == 1)).sum()),
        'tn': int(((y_true == 0) & (y_pred == 0)).sum()),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(rocauc),
        'avg_cost_per_sample': float(costs.sum() / len(y_true))
    }

# ==================== IDCSPA ALGORITHM (Section 3.3) ====================
class IDCSPA:
    """Instance-Dependent Cost-Sensitive Passive-Aggressive"""
    def __init__(self, C=1.0):
        self.C = C
        self.theta = None

    def _compute_cost_loss(self, y_pred, y_true, sum_insured):
        """Cost-sensitive loss from Eq. 13"""
        if y_true == -1 and y_pred == 1:  # FP
            return 100.0
        elif y_true == 1 and y_pred == -1:  # FN
            return sum_insured * 0.90
        return 0.0

    def fit(self, X, y, sum_insured, epochs=3):
        """Online learning with cost-sensitive updates"""
        n, d = X.shape
        self.theta = np.zeros(d)

        # Convert y to {-1, 1}
        y_signed = 2 * y - 1

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            for i in tqdm(idx, desc=f'IDCSPA Epoch {epoch+1}/{epochs}', leave=False):
                x_i = X[i]
                y_i = y_signed[i]
                w_i = sum_insured[i]

                # Prediction
                y_pred = np.sign(self.theta.dot(x_i))
                if y_pred == 0:
                    y_pred = -1

                # Cost-sensitive loss
                loss = self._compute_cost_loss(y_pred, y_i, w_i)

                if loss > 0:
                    # Update rule from Eq. 16
                    numerator = (y_pred - y_i) * self.theta.dot(x_i) + np.sqrt(loss)
                    denominator = np.linalg.norm((y_pred - y_i) * x_i) ** 2

                    if denominator > 0:
                        tau = min(self.C, numerator / denominator)
                        self.theta += tau * (y_i - y_pred) * x_i

        return self

    def predict(self, X):
        """Predict using sign(theta^T x)"""
        preds = np.sign(X.dot(self.theta))
        preds[preds == 0] = -1
        return ((preds + 1) / 2).astype(int)  # Convert to {0, 1}

    def decision_function(self, X):
        """Decision function for probability approximation"""
        return X.dot(self.theta)

class CSLB:
    """Cost-Sensitive Logistic Bandit with vectorized optimization + analytic gradient."""

    def __init__(self, alpha=1.0, lam=0.1, maxiter=100):
        self.alpha = alpha    # prior variance (used for initial cov)
        self.lam = lam        # L2 regularization (smooth)
        self.maxiter = maxiter
        self.theta_mean = None
        self.theta_cov = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _compute_threshold(self, sum_insured):
        # keep your formula (unchanged)
        a, b, c = 0.05, 10, 0.75
        C_FP = c * sum_insured + b
        C_TN = a * sum_insured
        C_TP = -b
        C_FN = -a * sum_insured
        # safe denominator guard
        denom = (C_FP - C_TN - C_TP + C_FN)
        return np.clip((C_FP - C_TN) / (denom if denom != 0 else 1e-9), 0.0, 1.0)

    def _obj_and_grad(self, theta, X, y, sum_insured):
        """
        Vectorized objective and gradient for average expected reward (AER).
        We use L2 regularization: lam * 0.5 * ||theta||^2
        Objective returns: scalar objective to minimize.
        Gradient returns: gradient vector (same shape as theta).
        """
        n = X.shape[0]
        z = X.dot(theta)
        p = self._sigmoid(z)                     # shape (n,)

        # coefficients per sample (see derivation)
        # For y==0: coeff = -(a+c)*w - b
        # For y==1: coeff = a*w - b
        a, b, c = 0.05, 10.0, 0.75
        w = sum_insured.astype(float)
        coeff = np.where(y == 0, (-(a + c) * w - b), (a * w - b))   # shape (n,)

        # total_reward = sum const_i + sum p_i * coeff_i
        # const_i = a*w  if y==0  OR -a*w if y==1  (we can include it but it cancels for gradient)
        const = np.where(y == 0, a * w, -a * w).sum()

        total_reward = const + (p * coeff).sum()
        # objective: - average reward + L2
        obj = -(total_reward / n) + 0.5 * self.lam * np.dot(theta, theta)

        # gradient: - (1/n) * sum coeff_i * p_i*(1-p_i) * x_i  + lam * theta
        s = coeff * p * (1 - p)                       # shape (n,)
        # vectorized grad: X.T @ s  (s multiplies each row of X)
        grad = -(X.T.dot(s) / n) + self.lam * theta   # shape (d,)

        return obj, grad

    def fit(self, X, y, sum_insured):
        n, d = X.shape
        if self.theta_mean is None:
            self.theta_mean = np.zeros(d)
            self.theta_cov = np.eye(d) * self.alpha

        # wrapper for minimize that returns objective scalar and gradient
        def fun(theta):
            obj, grad = self._obj_and_grad(theta, X, y, sum_insured)
            return obj, grad

        # call minimize with jac=True to supply gradient; limit iterations
        res = minimize(
            fun,
            self.theta_mean,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': self.maxiter, 'disp': False}
        )

        if res.success:
            self.theta_mean = res.x
        else:
            # if optimizer failed, keep previous mean but log a message (non-blocking)
            # this avoids "stuck" behaviour; it will continue with best-found x
            self.theta_mean = res.x
            # optionally: print("CSLB fit warning:", res.message)

        # Vectorized Hessian approximation for Laplace (X^T diag(p*(1-p)) X)
        eps = 1e-6
        z = X.dot(self.theta_mean)
        p = self._sigmoid(z)
        W = p * (1 - p)                        # shape (n,)
        # compute H = X.T @ (W[:,None] * X) + eps*I efficiently
        X_weighted = X * W[:, None]            # each row scaled
        H = X.T.dot(X_weighted) / max(1, n)    # normalized, keeps scale stable
        H += (eps + self.lam) * np.eye(d)      # add small diag + reg for invertibility

        # invert to get covariance (Laplace approximation)
        try:
            self.theta_cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.theta_cov = np.linalg.pinv(H)

        return self

    def predict(self, X, sum_insured):
        if self.theta_mean is None:
            raise ValueError("CSLB not fitted yet.")
        # Thompson sampling: sample one theta from posterior
        try:
            theta_sample = np.random.multivariate_normal(self.theta_mean, self.theta_cov)
        except Exception:
            theta_sample = self.theta_mean.copy()
        p = self._sigmoid(X.dot(theta_sample))
        thresholds = np.array([self._compute_threshold(s) for s in np.atleast_1d(sum_insured)])
        return (p >= thresholds).astype(int)

    def predict_proba(self, X):
        if self.theta_mean is None:
            raise ValueError("CSLB not fitted yet.")
        probs = self._sigmoid(X.dot(self.theta_mean))
        return np.vstack([1 - probs, probs]).T

# ==================== COMPARISON FUNCTION ====================
def compare_rl_algorithms(df, sample_size=20000, label_col='fraud_label', sum_col='sum_insured', random_state=42):
    """Compare RL algorithms with proper train-test split"""

    # Stratified sampling
    from sklearn.model_selection import train_test_split as split
    counts = df[label_col].value_counts()
    total = len(df)

    if sample_size >= total:
        df_sample = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        # Stratified sample
        sampled_parts = []
        for label in counts.index:
            group = df[df[label_col] == label]
            n_samples = int(sample_size * len(group) / total)
            sampled_parts.append(group.sample(n=min(n_samples, len(group)), random_state=random_state))
        df_sample = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Prepare features
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [label_col, sum_col]]

    X = df_sample[feature_cols].fillna(0).values
    y = df_sample[label_col].values
    sum_ins = df_sample[sum_col].values if sum_col in df_sample.columns else np.ones(len(df_sample)) * 1000

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Stratified train-test split
    X_train, X_test, y_train, y_test, sum_train, sum_test = train_test_split(
        X, y, sum_ins, test_size=0.30, random_state=random_state, stratify=y
    )

    results = {}

    # IDCSPA
    print("Training IDCSPA...")
    idcspa = IDCSPA(C=1.0).fit(X_train, y_train, sum_train, epochs=3)
    y_pred = idcspa.predict(X_test)
    results['IDCSPA'] = evaluate_cost_sensitive(y_test, y_pred, sum_test)

    # CSLB
    print("Training CSLB...")
    cslb = CSLB(alpha=1.0, lam=0.1).fit(X_train, y_train, sum_train)
    y_pred = cslb.predict(X_test, sum_test)
    results['CSLB'] = evaluate_cost_sensitive(y_test, y_pred, sum_test)

    # Results DataFrame
    comp = pd.DataFrame([
        {
            'Model': m,
            'Total Cost (£)': results[m]['total_cost'],
            'FN Cost (£)': results[m]['fn_cost'],
            'FP Cost (£)': results[m]['fp_cost'],
            'FN Count': results[m]['fn_count'],
            'FP Count': results[m]['fp_count'],
            'TP': results[m]['tp'],
            'TN': results[m]['tn'],
            'Precision': results[m]['precision'],
            'Recall': results[m]['recall'],
            'F1': results[m]['f1'],
            'ROC AUC': results[m]['roc_auc'],
            'Avg Cost/Sample (£)': results[m]['avg_cost_per_sample']
        } for m in results.keys()
    ]).sort_values('Total Cost (£)')

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Cost comparison
    ax = axes[0]
    models = comp['Model']
    costs = comp['Total Cost (£)']
    colors = ['#2ecc71' if c == costs.min() else '#e74c3c' if c == costs.max() else '#3498db' for c in costs]
    ax.barh(models, costs, color=colors)
    ax.set_xlabel('Total Cost (£)')
    ax.set_title('RL Algorithms: Total Cost Comparison', fontweight='bold')
    ax.invert_yaxis()
    for i, v in enumerate(costs):
        ax.text(v, i, f' £{v:,.0f}', va='center')

    # Metrics comparison
    ax = axes[1]
    x = np.arange(len(models))
    width = 0.25
    ax.bar(x - width, comp['Precision'], width, label='Precision', alpha=0.9)
    ax.bar(x, comp['Recall'], width, label='Recall', alpha=0.9)
    ax.bar(x + width, comp['F1'], width, label='F1', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('RL Algorithms: Performance Metrics', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return results, comp

results_rl, comparison_rl = compare_rl_algorithms(df, sample_size=20000)