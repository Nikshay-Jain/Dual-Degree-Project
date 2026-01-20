import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# ==================== STRATIFIED SAMPLING ====================
def stratified_sample(df, label_col='fraud_label', n=20000, random_state=42):
    """Stratified sampling with exact proportional allocation."""
    counts = df[label_col].value_counts().sort_index()
    total = counts.sum()
    
    if n >= total:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Largest remainder method for exact allocation
    proportions = counts / total
    raw_alloc = proportions * n
    floor_alloc = np.floor(raw_alloc).astype(int)
    remainder = raw_alloc - floor_alloc
    allocated = floor_alloc.copy()
    
    remaining = int(n - floor_alloc.sum())
    if remaining > 0:
        idx_sort = np.argsort(-remainder.values)
        for i in range(remaining):
            allocated[idx_sort[i]] += 1
    
    sampled_parts = []
    for label, k in zip(counts.index, allocated):
        group = df[df[label_col] == label]
        sampled_parts.append(group.sample(n=min(k, len(group)), random_state=random_state))
    
    return pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# ==================== COST FUNCTIONS ====================
def calculate_costs(sum_insured, y_true, y_pred):
    """Instance-dependent cost: FN=90% of sum_insured, FP=£100."""
    sum_insured = np.asarray(sum_insured)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    costs = np.zeros(len(y_true), dtype=float)
    
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    
    costs[fn_mask] = sum_insured[fn_mask] * 0.90
    costs[fp_mask] = 100.0
    return costs

def evaluate_cost_sensitive(y_true, y_pred, sum_insured):
    """Comprehensive cost-sensitive evaluation."""
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

# ==================== FEATURE PREPARATION ====================
def prepare_data(df, label_col='fraud_label', sum_col='sum_insured'):
    """Prepare features with robust fallback."""
    feature_template = [
        'tier', 'tier_premium_amt', 'occupants_count', 'paying_guest_no',
        'unoccupied_days', 'sum_insured', 'premium_with_tax',
        'risk_premium_with_credit_score', 'risk_premium_without_credit_score',
        'additional_coverage_amount', 'tier_limit_value', 'item_limit_value',
        'main_part_premium', 'accidental_damage_premium', 'outbuildings_premium',
        'APR_FLAG', 'previous_insurance_buildings', 'previous_insurance_cne',
        'property_eligibility', 'contents_claim_counts', 'building_claim_count',
        'flood_risk', 'storm_risk', 'other_natural_calamities_risk',
        'property_age', 'no_of_rooms', 'plumbing_elec_age',
        'smart_home', 'fire_safety_systems', 'security_systems',
        'crime_rate', 'no_of_claims_5yr', 'premium_payment_behaviour',
        'property_market_value', 'short_term_letting'
    ]
    categorical_template = ['cust_occupation', 'owner_details', 'property_type', 
                           'construction_material', 'roof_material', 'urban_rural', 'policy_product']
    
    feature_cols = [c for c in feature_template if c in df.columns and c != label_col]
    categorical_cols = [c for c in categorical_template if c in df.columns]
    
    if not feature_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in [label_col, sum_col]]
    
    df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    all_features = [c for c in df_enc.columns if (c in feature_cols) or any(cat in c for cat in categorical_cols)]
    
    if not all_features:
        all_features = [c for c in df_enc.select_dtypes(include=[np.number]).columns if c not in [label_col, sum_col]]
    
    X = df_enc[all_features].fillna(0).values
    y = df_enc[label_col].values
    sum_insured = df_enc[sum_col].values if sum_col in df_enc.columns else np.zeros(len(df_enc))
    
    return X, y, sum_insured, all_features

# ==================== COST-SENSITIVE CLASSIFIER ====================
class CostSensitiveClassifier:
    """Wrapper that optimizes probability threshold for minimum cost."""
    def __init__(self, base_clf):
        self.clf = base_clf
        self.opt_threshold = 0.5
    
    def fit(self, X, y, sum_insured=None):
        self.clf.fit(X, y)
        
        if sum_insured is not None and hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(X)[:, 1]
            thresholds = np.linspace(0.01, 0.99, 99)
            best_cost = np.inf
            best_t = 0.5
            
            for t in thresholds:
                y_pred = (proba >= t).astype(int)
                c = calculate_costs(sum_insured, y, y_pred).sum()
                if c < best_cost:
                    best_cost = c
                    best_t = t
            
            self.opt_threshold = best_t
        return self
    
    def predict(self, X):
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(X)[:, 1]
            return (proba >= self.opt_threshold).astype(int)
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)
        preds = self.clf.predict(X)
        return np.vstack([1-preds, preds]).T

# ==================== MAIN COMPARISON PIPELINE ====================
def compare_models(df, sample_size=20000, label_col='fraud_label', sum_col='sum_insured', random_state=42):
    """Train and compare cost-sensitive models."""
    
    # Stratified sampling
    df_sample = stratified_sample(df, label_col=label_col, n=sample_size, random_state=random_state)
    X, y, sum_ins, feature_names = prepare_data(df_sample, label_col=label_col, sum_col=sum_col)
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test, sum_train, sum_test = train_test_split(
        X, y, sum_ins, test_size=0.30, random_state=random_state, stratify=y
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    probs_store = {}
    
    # Model configurations
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=50, min_samples_leaf=20,
                                               class_weight='balanced', random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50,
                                               class_weight='balanced', random_state=random_state),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                scale_pos_weight=max(1, (y_train==0).sum() / max(1, (y_train==1).sum())),
                                random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Train and evaluate each model
    for name, base_clf in models.items():
        clf = CostSensitiveClassifier(base_clf).fit(X_train_s, y_train, sum_train)
        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]
        
        r = evaluate_cost_sensitive(y_test, y_pred, sum_test)
        r['proba'] = y_proba
        results[name] = r
        probs_store[name] = (y_test, y_proba)
    
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
    
    # ==================== VISUALIZATIONS ====================
    
    # Figure 1: Cost Analysis
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total cost comparison
    ax = axes[0, 0]
    models_list = comp['Model']
    costs = comp['Total Cost (£)']
    colors = ['#2ecc71' if c == costs.min() else '#e74c3c' if c == costs.max() else '#3498db' for c in costs]
    ax.barh(models_list, costs, color=colors)
    ax.set_xlabel('Total Cost (£)')
    ax.set_title('Total Cost Comparison (lower = better)', fontweight='bold')
    ax.invert_yaxis()
    for i, v in enumerate(costs):
        ax.text(v, i, f' £{v:,.0f}', va='center')
    
    # Cost breakdown
    ax = axes[0, 1]
    x = np.arange(len(models_list))
    width = 0.35
    ax.bar(x - width/2, comp['FN Cost (£)'], width, label='FN Cost (missed fraud)', alpha=0.9, color='#e74c3c')
    ax.bar(x + width/2, comp['FP Cost (£)'], width, label='FP Cost (false accusation)', alpha=0.9, color='#f39c12')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=45, ha='right')
    ax.set_ylabel('Cost (£)')
    ax.set_title('Cost Breakdown by Error Type', fontweight='bold')
    ax.legend()
    
    # Precision-Recall tradeoff
    ax = axes[1, 0]
    sc = ax.scatter(comp['Recall'], comp['Precision'], s=200, c=comp['Total Cost (£)'], 
                    cmap='RdYlGn_r', edgecolors='k', linewidths=2)
    for i, m in enumerate(models_list):
        ax.annotate(m, (comp['Recall'].iloc[i], comp['Precision'].iloc[i]), 
                   xytext=(6, 4), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Tradeoff (color = total cost)', fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Total Cost (£)')
    ax.grid(alpha=0.3)
    
    # Error counts
    ax = axes[1, 1]
    ax.bar(x - width/2, comp['FN Count'], width, label='False Negatives', alpha=0.9, color='#e74c3c')
    ax.bar(x + width/2, comp['FP Count'], width, label='False Positives', alpha=0.9, color='#f39c12')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Error Counts by Model', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: PR and ROC Curves
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precision-Recall curves
    ax = axes[0]
    for m, (y_t, y_p) in probs_store.items():
        try:
            p, r, _ = precision_recall_curve(y_t, y_p)
            ax.plot(r, p, label=f"{m} (F1={results[m]['f1']:.3f})", linewidth=2)
        except:
            continue
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ROC curves
    ax = axes[1]
    for m, (y_t, y_p) in probs_store.items():
        try:
            fpr, tpr, _ = roc_curve(y_t, y_p)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{m} (AUC={roc_auc:.3f})", linewidth=2)
        except:
            continue
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Confusion Matrices
    n = len(results)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig3, axes = plt.subplots(rows, cols, figsize=(10, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, m in enumerate(results.keys()):
        y_t = probs_store[m][0]
        y_pred = (results[m]['proba'] >= 0.5).astype(int)
        cm = confusion_matrix(y_t, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                   xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
        axes[i].set_title(f'{m}\n(Cost: £{results[m]["total_cost"]:,.0f})', fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    for j in range(i+1, len(axes)):
        fig3.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    return results, comp

results, comparison_df = compare_models(df, sample_size=20000)