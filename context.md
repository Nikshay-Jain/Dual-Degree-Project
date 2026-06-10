# MASTER CONTEXT FILE: Actuarial Active Learning via SEDR

**Project:** Economically Optimized Pure Premium Pricing via Stratified Expected Deviance Reduction (SEDR) in Active Learning.
**Domain:** Property & Casualty (P&C) Insurance Data Science / Actuarial Mathematics.
**Current Phase:** Phase 2 (Phase 1 was Fraud Detection, which has been explicitly pivoted away from).
**Target Variable:** Continuous expected payout (Pure Premium = Claim Amount / Exposure).

---

## 1. The Core Actuarial Problem & Mathematical Traps

### 1.1 The Distribution

Insurance pure premium follows a **Compound Poisson-Gamma (Tweedie)** distribution ($1 < p < 2$).

* **Zero-Inflated:** $\ge 90\%$ of policyholders have zero claims (point mass at 0).
* **Heavy-Tailed:** The remaining $5-10\%$ of claims form a continuous, right-skewed severity tail.

### 1.2 The Mean-Variance Coupling Trap

In standard Active Learning (AL), algorithms query the "most uncertain" data points, usually measured by variance. However, in a Tweedie GLM, variance is deterministically coupled to the predicted mean ($\mu$) via the power law:


$$Var(Y) = \phi \mu^p$$

* **The Failure:** If a naive AL algorithm queries by highest variance, it conflates *model ignorance* with *inherent claim severity*. It will exclusively sample the most expensive, catastrophic crashes (the heavy tail) and completely ignore normal, safe drivers.
* **The Consequence (Calibration Collapse):** The model's internal dispersion parameter ($\phi$) artificially explodes. Because the implied Poisson frequency is $\lambda = \mu^{2-p} / (\phi(2-p))$, this explosion forces the frequency parameter ($\lambda$) to collapse toward zero. The model loses its ability to price safe drivers and destroys its Zero-Mass Calibration.

---

## 2. The Proposed Solution: SEDR Architecture

To solve this, we engineered **Stratified Expected Deviance Reduction (SEDR)**. It rests on two pillars:

1. **Epistemic Decoupling (Pure Tree Disagreement):** To stop the model from chasing "loud" claims, we measure uncertainty purely as the variance of predictions across LightGBM's sequential tree-growth checkpoints, extracted in the *raw log-link space* ($F(x)$) before the exponential transformation. This isolates genuine model confusion (Epistemic uncertainty) from claim magnitude (Aleatoric noise).
2. **Stratified Risk Quotas:** The unlabeled pool is partitioned into 5 risk quintiles based on expected $\mu$. To enforce a biologically accurate "diet" of data, a strict quota is applied: **30% of the query budget goes to the safest tier (Quintile 1, the zero-mass), and the remaining 70% is distributed evenly across the upper 4 risk strata.**

---

## 3. Mathematical Evaluators & Metrics Developed

### 3.1 Tweedie Evaluator

* **Tweedie Deviance:** $d(y, \mu) = 2 \left( \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right)$. Weighted by policy `Exposure`.
* **Zero-Mass Calibration Error (ZMCE):** Measures how accurately the model predicts the true $P(Y=0)$.
* *Crucial Fix:* The expected frequency must scale linearly with exposure. $\lambda_i = w_i \cdot \mu_i^{2-p} / (\phi(2-p))$.
* *Leakage Fix:* The dispersion parameter ($\phi$) must be estimated *strictly* on the `y_train` set, not the holdout set, to prevent oracle leakage.



### 3.2 Economic Evaluator (Lorenz Curve & Gini)

* **Purpose:** Measures pure rank-ordering and price discrimination.
* **Crucial Fix (Time-Blindness):** Standard Gini sorts by predicted risk but uses raw row counts for the X-axis. We modified it to an **Exposure-Weighted Gini**. The cumulative population X-axis is explicitly `np.cumsum(exposure_sorted)` to align with real-world actuarial utility.

### 3.3 Business Impact Evaluator

Derived from an audit of a secondary "Dual Model" notebook, we rejected the Dual Model architecture but adopted its real-world business metrics:

1. **Financial Bias (%):** $(Total\ Predicted - Total\ Actual) / Total\ Actual$. Measures portfolio reserving error.
2. **Spearman Rank Correlation ($\rho$):** Non-parametric monotonicity check.
3. **Catastrophic Threat ID:** Averages the global predicted percentile rank of the actual top 5% most severe claims (evaluates reinsurance stop-loss prioritization).

---

## 4. The 4 Active Learning Pipelines

1. **Random (NSL):** The baseline. Preserves population structure but has terrible learning efficiency.
2. **Naive Variance:** Ranks strictly by predicted $\mu$ (time-blind, no exposure scaling). Represents the industry failure mode.
3. **Epistemic-Only:** Queries purely by tree-disagreement without strata.
4. **SEDR:** Our novel framework (Tree Disagreement + Stratification).
5. **One-Shot Baseline:** A flat-line model trained on the terminal data budget (Seed + (Iterations * QuerySize)) via random sampling, serving as the visual threshold AL models must cross.

---

## 5. Empirical Proofs: The "Dual Failure" Validation

During our testing, the mathematical hypotheses were perfectly proven:

* **The Severity Trap (Naive Variance):** Gini collapsed to ~0.14. By over-sampling the tail, it destroyed its ability to rank safe drivers and collapsed the ZMCE via parameter distortion.
* **The Frequency Trap (Epistemic-Only):** ZMCE was perfect, but Deviance spiked catastrophically. By hunting pure model confusion without boundaries, it got stuck sampling micro-noise between safe drivers and was completely blind to severe claims.
* **The Victory (SEDR):** Achieved optimal Deviance (halving the error of Random) while maintaining the structural Gini index (0.28+) and keeping ZMCE anchored to near zero.

---

## 6. The Synthetic Oracle Suite (Task 3.2)

To prove SEDR reconstructs actuarial ground truth faster than other methods, we built a synthetic universe.

* **The Exposure Invariant Flaw & Fix:** Initially, our synthetic generator scaled the overall mean by exposure but left dispersion constant. This violated the GLM invariant $\phi_Y = \phi w^{1-p}$, causing crash severity to shrink on shorter policies.
* **The Correction:** We explicitly hardcoded $\lambda_{true} = w \cdot \frac{\mu^{2-p}}{\phi(2-p)}$ (frequency scales linearly with exposure) and $\theta_{true} = \phi(p-1)\mu^{p-1}$ (severity is independent of exposure).
* **Result:** We track Mean Absolute Error (MAE) against the hidden $\mu_{true}$. SEDR smoothly reconstructs the generative function, while Naive Variance flatlines due to noise-chasing.

---

## 7. Optimizing Tweedie Power ($p$): A Chronicle of Mathematical Corrections

This was the most complex mathematical hurdle. Setting $p$ defines the entire loss function landscape.

* **Failed Path 1 (Deviance Comparison):** We initially considered tuning $p$ by picking the lowest Deviance. *Why it failed:* The Tweedie formula has $p$ in exponents and denominators. Deviance values are on completely different scales for $p=1.1$ vs $p=1.9$. It is mathematically invalid to compare them iteratively.
* **Failed Path 2 (Gini / MAE Composite Score):** We tried normalizing Gini and MAE and combining them. *Why it failed:* First, it leaked data by tuning on the AL pool. Second, $p=1.05$ generated a massive, outlier MAE (205.48), which warped the normalization scale and rejected actually optimal models based on arbitrary hardcoded bounds.
* **Failed Path 3 (Pearson Residual Independence):** We tried correlating predicted means with squared Pearson residuals to find the $p$ that makes them orthogonal. *Why it failed:* As $p \to 1.95$, the denominator ($\mu_i^p$) exponentially inflates, artificially squashing the variance of residuals and forcing the correlation to zero. It just found the mathematical boundary, not the true power.
* **THE FINAL SOLUTION (Empirical Variance Function Estimation):** We abandoned iterative loop-guessing. In a Tweedie GLM, $Var(Y) = \phi \mu^p \implies \log(Var) = \log(\phi) + p \log(\mu)$.
*Method:* We use 5-fold OOF predictions purely on the AL `seed` set to group drivers into 10 risk deciles. We calculate the empirical log-mean and log-variance for each bucket. The slope of a linear regression between them is the mathematically true $p$ embedded in the dataset. (Loop-free, leak-free, mathematically unassailable).

---

## 8. Why We Explicitly REJECTED Optimizing for MAE and RMSE

A critical analysis was conducted on whether the AL models should optimize for MAE or RMSE. The answer was definitively **NO**.

* **The MAE Trap (Zero-Collapse):** MAE optimizes for the *median*. Because >90% of claims are zero, minimizing MAE forces the model to predict $\$0$ for almost everyone, which bankrupts the insurer.
* **The RMSE Trap (Tail Explosion):** RMSE assumes symmetric, constant variance. It heavily penalizes outliers. Because a $\$100k$ claim creates a squared penalty of $10^9$, the model artificially overprices safe drivers to hedge against rare tail risks, destroying competitive pricing.
* **Implementation:** We added `wmae` and `wrmse` strictly as "tracking metrics" to the evaluator so we can graph them and prove their failure to the thesis committee, but the active learning acquisition remains strictly governed by Tweedie Deviance.

---