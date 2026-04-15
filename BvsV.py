"""
MAT 4373 — Mathematical Machine Learning
Group Project 12: Double Descent — Computational Experiments
=========================================================

Reproduces the double descent phenomenon from:
  Belkin, Hsu, Ma & Mandal (2019). Reconciling modern machine learning
  and the bias–variance trade-off. PNAS, 116(32), 15849–15854.

Experiments:
  1. Polynomial regression — training/test error vs. degree (with B-V decomposition)
  2. Gaussian feature model — test MSE vs. p/n ratio (ridgeless vs. ridge)
  3. Regularisation sweep — effect of ridge penalty α on the double descent curve

Requirements:
  pip install numpy matplotlib scikit-learn scipy pillow

Run:
  python double_descent_experiments.py

Output (saved to current directory):
  fig1_polynomial_bvdd.png
  fig2_gaussian_double_descent.png
  fig3_regularisation_effect.png
  fig4_schematic_double_descent.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COLORS = {
    'train':         '#2563EB',
    'test':          '#DC2626',
    'bias2':         '#7C3AED',
    'variance':      '#059669',
    'interpolation': '#B45309',
    'noise':         '#6B7280',
    'ridge':         '#0891B2',
}


# polynomial regression part
def generate_data(n, sigma=0.5, seed=None):     # data gen process using y = sin(3πx) as true function
    """
    Generate samples from  y = sin(3πx) + N(0, σ²),  x ~ Uniform(-1, 1).
    Returns (X, y) with X shaped (n, 1).
    """
    rng = np.random.default_rng(seed)
    x   = rng.uniform(-1, 1, n)
    y   = np.sin(3 * np.pi * x) + sigma * rng.standard_normal(n)
    return x.reshape(-1, 1), y


def fit_poly(degree, X_train, y_train): 
    n = len(y_train)
    poly  = PolynomialFeatures(degree, include_bias=True)
    X_tr_p = poly.fit_transform(X_train)
    p = X_tr_p.shape[1]

    if p <= n:  # underparametrised regime
        reg = LinearRegression(fit_intercept=False).fit(X_tr_p, y_train)
    else:       # overparametrised regime
        reg = Ridge(alpha=1e-10, fit_intercept=False).fit(X_tr_p, y_train)
    class _PolyModel:
        def predict(self, X):
            return reg.predict(poly.transform(X))

    return _PolyModel()


def poly_train_test_errors(degrees, n_train=20, n_test=500, n_seeds=150, sigma=0.5):
    X_test, y_test = generate_data(n_test, sigma=sigma, seed=999)

    train_errors, test_errors = [], []
    for deg in degrees:
        tr_list, te_list = [], []
        for s in range(n_seeds):
            X_tr, y_tr = generate_data(n_train, sigma=sigma, seed=s)
            try:
                model = fit_poly(deg, X_tr, y_tr)
                tr_list.append(np.mean((model.predict(X_tr) - y_tr) ** 2))
                te_list.append(np.mean((model.predict(X_test) - y_test) ** 2))
            except Exception:
                pass
        train_errors.append(np.nanmean(tr_list))
        test_errors.append(np.nanmean(te_list))

    return np.array(train_errors), np.array(test_errors) # approx risk


def bias_variance_decomposition(degrees, n_train=20, n_test=300, n_seeds=200, sigma=0.5):

    X_test, _ = generate_data(n_test, sigma=0, seed=999)
    f_star     = np.sin(3 * np.pi * X_test.ravel())

    bias2_list, var_list, noise_list = [], [], []
    for deg in degrees:
        preds = []
        for s in range(n_seeds):
            X_tr, y_tr = generate_data(n_train, sigma=sigma, seed=s)
            try:
                model = fit_poly(deg, X_tr, y_tr)
                preds.append(model.predict(X_test))
            except Exception:
                pass
        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)
        bias2_list.append(np.mean((mean_pred - f_star) ** 2))
        var_list.append(np.mean(np.var(preds, axis=0)))
        noise_list.append(sigma ** 2)

    return np.array(bias2_list), np.array(var_list), np.array(noise_list)


#gaussian feature model section

def gaussian_feature_model_test_mse(p_values, n_train=50, n_test=500, n_seeds=80, sigma=1.0, alpha=0.0):
    """
    Gaussian feature model:
      y = X w* + ε,  X_ij ~ N(0, 1/p),  ε ~ N(0, σ²)
      w* is sparse: w*_i = 1/min(5,p) for i < 5, else 0.
    """

    rng = np.random.default_rng(0)
    test_errors = []

    for p in p_values:
        w_star = np.zeros(p)
        w_star[:min(5, p)] = 1.0 / min(5, p)

        # Fixed test set for p
        X_test = rng.standard_normal((n_test, p)) / np.sqrt(p)
        y_test = X_test @ w_star + sigma * rng.standard_normal(n_test)

        err_list = []
        for s in range(n_seeds):
            rng2 = np.random.default_rng(s + 100)
            X_tr = rng2.standard_normal((n_train, p)) / np.sqrt(p)
            y_tr = X_tr @ w_star + sigma * rng2.standard_normal(n_train)
            try:
                a = max(alpha, 1e-10)   # never exactly 0 for numerical stability
                reg = Ridge(alpha=a, fit_intercept=False).fit(X_tr, y_tr)
                err_list.append(np.mean((reg.predict(X_test) - y_test) ** 2))
            except Exception:
                pass
        test_errors.append(np.nanmean(err_list))

    return np.array(test_errors)


# exp1 setup
DEGREES   = list(range(1, 36))  # degrees 1–19 are underparametrised, degree 19/20 is near the interpolation threshold, and degrees 21–35 are overparametrised
N_TRAIN   = 20
SIGMA     = 0.5
INTERP_DEG = 20    # p = d+1 = 20 ≈ n at degree 19

print("Running Experiment 1a: polynomial train/test error (150 seeds)...")
train_err, test_err = poly_train_test_errors(
    DEGREES, n_train=N_TRAIN, n_seeds=150, sigma=SIGMA
)

print("Running Experiment 1b: bias-variance decomposition (200 seeds)...")
bias2, variance, noise = bias_variance_decomposition(
    DEGREES, n_train=N_TRAIN, n_seeds=200, sigma=SIGMA
)

# exp2 and exp3
N_TRAIN_GAUSS = 50
P_VALUES = sorted(set(
    list(range(2, 8)) +
    list(range(8, 48, 2)) +
    list(range(48, 55)) +
    list(range(55, 201, 5))
))
RATIO = np.array(P_VALUES) / N_TRAIN_GAUSS

print("Running Experiment 2: Gaussian feature model — ridgeless vs ridge (80 seeds each)...")
dd_ridgeless = gaussian_feature_model_test_mse(P_VALUES, n_train=N_TRAIN_GAUSS, n_seeds=80, sigma=1.0, alpha=0.0)
dd_ridge_1   = gaussian_feature_model_test_mse(P_VALUES, n_train=N_TRAIN_GAUSS, n_seeds=80, sigma=1.0, alpha=1.0)

print("Running Experiment 3: regularisation sweep (80 seeds each)...")
ALPHAS       = [0.0, 0.1, 1.0, 10.0]
ALPHA_COLORS = ['#DC2626', '#F59E0B', '#2563EB', '#059669']
ALPHA_LABELS = ['α → 0 (ridgeless)', 'α = 0.1', 'α = 1.0', 'α = 10.0']
dd_by_alpha  = [
    gaussian_feature_model_test_mse(P_VALUES, n_train=N_TRAIN_GAUSS, n_seeds=80, sigma=1.0, alpha=a)
    for a in ALPHAS
]

print("All experiments complete. Generating figures...")


# figure 2
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(
    "EXPERIMENT 1  ·  Polynomial Regression  (n = 20, σ = 0.5): "
    "Train/Test Error and Bias–Variance Decomposition",
    fontsize=11, fontweight='bold', y=1.01
)

# train vs test error
ax = axes[0]
ax.axvline(INTERP_DEG, color=COLORS['interpolation'], lw=1.5, ls='--', alpha=0.7,
           label='Interpolation threshold')
ax.plot(DEGREES, train_err, color=COLORS['train'],  lw=2, marker='o', ms=3.5, label='Training MSE')
ax.plot(DEGREES, test_err,  color=COLORS['test'],   lw=2, marker='s', ms=3.5, label='Test MSE (risk estimate)')
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('Mean Squared Error')
ax.set_title('(a)  Training vs. Test Error')
ax.set_ylim(0, min(test_err.max() * 1.1, 4.0))
ax.legend(frameon=False, fontsize=9)
ax.annotate(
    'Interpolation\nthreshold\n(p ≈ n)',
    xy=(INTERP_DEG, test_err[INTERP_DEG - 1]),
    xytext=(INTERP_DEG + 2.5, test_err[INTERP_DEG - 1] + 0.5),
    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8, color='gray'
)

# bias, variance, and noise
ax2 = axes[1]
ax2.axvline(INTERP_DEG, color=COLORS['interpolation'], lw=1.5, ls='--', alpha=0.7,
            label='Interpolation threshold')
ax2.plot(DEGREES, bias2,            color=COLORS['bias2'],    lw=2, marker='o', ms=3.5, label='Bias²')
ax2.plot(DEGREES, variance,         color=COLORS['variance'], lw=2, marker='s', ms=3.5, label='Variance')
ax2.axhline(noise[0], color=COLORS['noise'], lw=1.5, ls=':', label=f'Irreducible noise (σ²={noise[0]:.2f})')
ax2.plot(DEGREES, bias2 + variance + noise, color='black', lw=1.5, ls='-.', alpha=0.7, label='Bias²+Var+Noise (total risk)')
ax2.set_xlabel('Polynomial degree')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('(b)  Bias², Variance, and Irreducible Noise')
ax2.set_ylim(0, min((bias2 + variance + noise).max() * 1.1, 5.0))
ax2.legend(frameon=False, fontsize=8.5)

plt.tight_layout()
plt.savefig('fig1_polynomial_bvdd.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Saved: fig1_polynomial_bvdd.png")


# figure 3 - gaussian feature model

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.suptitle(
    "EXPERIMENT 2  ·  Double Descent in the Gaussian Feature Model\n"
    "(n = 50, σ = 1, varying p;  reproducing Belkin et al. 2019 / Hastie et al. 2022 style)",
    fontsize=11, fontweight='bold'
)

ax.axvline(1.0, color=COLORS['interpolation'], lw=2, ls='--', alpha=0.8, zorder=3,
           label='Interpolation threshold (p/n = 1)')
ax.plot(RATIO, dd_ridgeless, color=COLORS['test'],  lw=2.2, marker='o', ms=3,
        label='Ridgeless (α → 0): minimum-norm interpolation')
ax.plot(RATIO, dd_ridge_1,   color=COLORS['ridge'], lw=2.2, marker='s', ms=3, ls='--',
        label='Ridge (α = 1.0): explicit regularisation')

ax.axvspan(0, 1,           alpha=0.07, color=COLORS['bias2'],    label='Under-parametrised regime')
ax.axvspan(1, max(RATIO),  alpha=0.07, color=COLORS['variance'], label='Over-parametrised regime')

# divergence peak
mask = (RATIO > 0.85) & (RATIO < 1.15)
peak_idx = np.argmax(dd_ridgeless[mask])
peak_x   = RATIO[mask][peak_idx]
peak_y   = dd_ridgeless[mask][peak_idx]
ax.annotate(
    'Peak at p/n ≈ 1\n(interpolation threshold)',
    xy=(peak_x, peak_y), xytext=(peak_x + 0.35, peak_y * 0.80),
    arrowprops=dict(arrowstyle='->', color='black'), fontsize=8.5, ha='left'
)

ax.set_xlabel('p / n   (number of parameters / number of training samples)', fontsize=10)
ax.set_ylabel('Test MSE', fontsize=10)
ax.set_title('Ridgeless interpolation diverges at p = n, then descends', fontsize=9)
ax.set_xlim(left=0)
ax.set_ylim(0, min(np.percentile(dd_ridgeless, 97) * 1.5, 20))
ax.legend(frameon=False, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('fig2_gaussian_double_descent.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Saved: fig2_gaussian_double_descent.png")


# regularization sweep
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.suptitle(
    "EXPERIMENT 3  ·  Implicit vs. Explicit Regularisation in Double Descent\n"
    "(Effect of Ridge penalty α on the double descent curve;  n = 50, σ = 1)",
    fontsize=11, fontweight='bold'
)

for dd, a, col, lab in zip(dd_by_alpha, ALPHAS, ALPHA_COLORS, ALPHA_LABELS):
    ls = '-' if a == 0.0 else '--'
    ax.plot(RATIO, dd, color=col, lw=2, ls=ls, label=lab)

ax.axvline(1.0, color='gray', lw=1.5, ls=':', alpha=0.8)
ax.text(1.03, 13, 'p = n', fontsize=9, color='gray')
ax.set_xlabel('p / n   (number of parameters / number of training samples)', fontsize=10)
ax.set_ylabel('Test MSE', fontsize=10)
ax.set_title('Stronger regularisation dampens the divergence peak', fontsize=9)
ax.set_xlim(left=0)
ax.set_ylim(0, 15)
ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig('fig3_regularisation_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_regularisation_effect.png")


# double descent schema
def _schematic_curve(x):
    """Piecewise smooth schematic of a double descent risk curve."""
    left         = 2.0 + 1.5 * (x - 0.8) ** 2
    peak         = 6.0 * np.exp(-15 * (x - 1.0) ** 2)
    right_descent = 1.3 * np.exp(-0.9 * (x - 1.0)) + 0.9
    return np.where(x < 1.0, left + peak, right_descent + peak)

x = np.linspace(0.05, 3.5, 400)
y_dd = _schematic_curve(x)
y_u  = np.where(x < 1.2, 2.0 + 1.5 * (x - 0.8) ** 2, np.nan)   # classic U-shape only

fig, ax = plt.subplots(figsize=(10, 4.2))
fig.suptitle(
    "Figure 1  ·  Schematic: Classical U-Shape vs. Double Descent Risk Curve",
    fontsize=11, fontweight='bold'
)

ax.fill_betweenx([0, 9], 0,  1, alpha=0.08, color='#7C3AED', label='Classical (under-parametrised)')
ax.fill_betweenx([0, 9], 1, 3.5, alpha=0.08, color='#059669', label='Modern (over-parametrised)')
ax.axvline(1.0, color='#B45309', lw=2, ls='--', label='Interpolation threshold (p = n)')

ax.plot(x, y_u,  color='#94A3B8', lw=2, ls=':', label='Classical U-shape (bias-var tradeoff only)')
ax.plot(x, y_dd, color='#DC2626', lw=2.5,        label='Double descent risk curve')

ax.annotate('Bias² dominates\n(high bias, low var)', xy=(0.4, 5.5), fontsize=8.5,
            color='#7C3AED', ha='center')
ax.annotate('Variance\ndominates', xy=(0.85, 4.0), fontsize=8.5, color='#7C3AED', ha='center')
ax.annotate('Peak: interpolation\nthreshold divergence', xy=(1.4, 7.0), fontsize=8.5,
            color='#B45309', ha='center', va='bottom')
ax.annotate('Second descent\n(implicit regularisation)', xy=(2.5, 2.2), fontsize=8.5,
            color='#059669', ha='center',
            arrowprops=dict(arrowstyle='->', color='#059669'),
            xytext=(2.5, 3.5))

ax.set_xlabel('Model complexity  (p/n  or  polynomial degree / n)', fontsize=10)
ax.set_ylabel('Test Risk (MSE)', fontsize=10)
ax.set_xlim(0.05, 3.5)
ax.set_ylim(0, 9)
ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ax.set_xticklabels(['Low\ncomplexity', 'p = n\n(interp.)', '1.5×', '2×', '2.5×', '3×'])
ax.legend(frameon=False, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('fig4_schematic_double_descent.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Saved: fig4_schematic_double_descent.png")

print("\nDone! All 4 figures saved to the current working directory.")