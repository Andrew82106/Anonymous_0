import numpy as np
from scipy.stats import skew, kurtosis, pearsonr, spearmanr, chi2_contingency, entropy, fisher_exact
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, mutual_info_score
from sklearn.feature_selection import mutual_info_regression

class StatTranslator:
    def __init__(self):
        pass

    def analyze(self, X, Y):
        """
        Analyze the statistical relationship between X and Y to infer causal direction.
        X, Y: 1D numpy arrays
        Returns: A dictionary containing statistical features.
        """
        X = np.array(X).flatten()
        Y = np.array(Y).flatten()
        
        stats = {}
        
        # 1. Univariate Statistics
        stats['len'] = len(X)
        
        # Check if variables are discrete
        is_discrete_x = self._is_discrete(X)
        is_discrete_y = self._is_discrete(Y)
        stats['is_discrete'] = is_discrete_x and is_discrete_y
        
        if stats['is_discrete']:
            # Discrete Analysis
            stats['x_unique'] = len(np.unique(X))
            stats['y_unique'] = len(np.unique(Y))
            
            # Marginal Entropy (Entropy Accumulation Hypothesis: Cause has lower entropy than Effect)
            def calc_entropy(arr):
                _, counts = np.unique(arr, return_counts=True)
                return entropy(counts / len(arr), base=2)
            
            stats['x_entropy'] = calc_entropy(X)
            stats['y_entropy'] = calc_entropy(Y)
            
            # Conditional Entropy & Accuracy Analysis
            stats['dir_ab'] = self._evaluate_direction_discrete(X, Y)
            stats['dir_ba'] = self._evaluate_direction_discrete(Y, X)
            
            # Calculate Correlation (Cramer's V or similar, but Pearson/Spearman ok for ordered)
            # We keep Pearson/Spearman as general indicators even for discrete
            stats['pearson_corr'], _ = pearsonr(X, Y) if np.issubdtype(X.dtype, np.number) and np.issubdtype(Y.dtype, np.number) else (0, 0)
            
        else:
            # Continuous Analysis (Existing logic)
            stats['x_skew'] = skew(X)
            stats['x_kurt'] = kurtosis(X)
            stats['y_skew'] = skew(Y)
            stats['y_kurt'] = kurtosis(Y)
            
            corr_p, _ = pearsonr(X, Y)
            corr_s, _ = spearmanr(X, Y)
            stats['pearson_corr'] = corr_p
            stats['spearman_corr'] = corr_s
            
            stats['dir_ab'] = self._evaluate_direction(X, Y)
            stats['dir_ba'] = self._evaluate_direction(Y, X)
        
        return stats

    def _is_discrete(self, arr, threshold=15):
        """Heuristic to check if a variable is discrete"""
        # If string/object, it's discrete
        if arr.dtype.kind in {'U', 'S', 'O'}:
            return True
        # If numeric but few unique values relative to length (or absolute low count)
        unique_count = len(np.unique(arr))
        return unique_count < threshold or (unique_count / len(arr) < 0.05 and unique_count < 100)

    def _evaluate_direction_discrete(self, predictor, target):
        """
        Evaluate causal direction for discrete variables using classification.
        """
        # Encode labels
        le_p = LabelEncoder()
        X_enc = le_p.fit_transform(predictor).reshape(-1, 1)
        le_t = LabelEncoder()
        y_enc = le_t.fit_transform(target)
        
        # 1. Fit Classifier (Logistic Regression)
        # Using multiclass if needed
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
        clf.fit(X_enc, y_enc)
        
        # 2. Predict
        y_pred = clf.predict(X_enc)
        y_prob = clf.predict_proba(X_enc)
        
        # 3. Metrics
        acc = accuracy_score(y_enc, y_pred)
        try:
            ll = log_loss(y_enc, y_prob)
        except:
            ll = np.nan
            
        # 4. Conditional Entropy H(Y|X)
        # H(Y|X) = sum p(x) * H(Y|X=x)
        total_entropy = 0
        unique_x, counts_x = np.unique(predictor, return_counts=True)
        total_count = len(predictor)
        
        for x_val, count in zip(unique_x, counts_x):
            y_given_x = target[predictor == x_val]
            # Calculate entropy of Y given X=x
            _, counts_y_given_x = np.unique(y_given_x, return_counts=True)
            probs_y_given_x = counts_y_given_x / count
            h_y_given_x = entropy(probs_y_given_x, base=2)
            
            weight = count / total_count
            total_entropy += weight * h_y_given_x
            
        # 5. Independence of Errors (Residuals)
        # For discrete, "error" means Y != Y_pred.
        # Ideally, the error rate should be independent of X.
        errors = (y_enc != y_pred).astype(int)
        
        # Chi-square test of independence between X and Error
        # Construct contingency table: X values vs Error (0 or 1)
        # Note: if X has many values, this might be sparse.
        if len(unique_x) < 20:
            contingency = np.array([
                [np.sum((X_enc.flatten() == i) & (errors == 0)) for i in range(len(unique_x))],
                [np.sum((X_enc.flatten() == i) & (errors == 1)) for i in range(len(unique_x))]
            ])
            # Add minimal smoothing to avoid zero division in chi2
            contingency = contingency + 0.1
            chi2, p_val, _, _ = chi2_contingency(contingency)
            # High p-value means independence (Good)
            # Low p-value means dependence (Bad, error rate depends on X)
        else:
            p_val = 0.5 # Default if too many categories
        
        # 6. Mutual Information between X and Y (for Sprinkler-type cases)
        # Higher MI = Stronger association (helps when entropy signals are ambiguous)
        mi_xy = mutual_info_score(predictor, target)
            
        return {
            'accuracy': acc,
            'log_loss': ll,
            'conditional_entropy': total_entropy,
            'error_independence_p': p_val,
            'mutual_information': mi_xy,  # NEW: Added for better discrimination
            'model_type': 'logistic_classifier'
        }

    def _evaluate_direction(self, predictor, target):
        """
        Fit target = f(predictor) + noise
        Test both linear and non-linear (polynomial) models
        Return stats about the residuals and model fit.
        """
        X2 = predictor.reshape(-1, 1)
        y2 = target.reshape(-1, 1)
        
        # 1. Linear Fit
        model_linear = LinearRegression().fit(X2, y2)
        residuals_linear = y2 - model_linear.predict(X2)
        residuals_linear = residuals_linear.flatten()
        r2_linear = model_linear.score(X2, y2)
        
        # 2. Non-linear Fit (Polynomial degree 2)
        model_poly2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model_poly2.fit(X2, y2.ravel())
        residuals_poly2 = y2.ravel() - model_poly2.predict(X2)
        r2_poly2 = model_poly2.score(X2, y2.ravel())
        
        # 3. Non-linear Fit (Polynomial degree 3)
        model_poly3 = make_pipeline(PolynomialFeatures(3), LinearRegression())
        model_poly3.fit(X2, y2.ravel())
        residuals_poly3 = y2.ravel() - model_poly3.predict(X2)
        r2_poly3 = model_poly3.score(X2, y2.ravel())
        
        # Select best model based on R2 improvement
        r2_improvement_poly2 = r2_poly2 - r2_linear
        r2_improvement_poly3 = r2_poly3 - r2_poly2
        
        # Use the best model's residuals
        if r2_improvement_poly3 > 0.05:  # Significant improvement with degree 3
            best_residuals = residuals_poly3
            best_r2 = r2_poly3
            best_model_type = 'polynomial_3'
        elif r2_improvement_poly2 > 0.05:  # Significant improvement with degree 2
            best_residuals = residuals_poly2
            best_r2 = r2_poly2
            best_model_type = 'polynomial_2'
        else:
            best_residuals = residuals_linear
            best_r2 = r2_linear
            best_model_type = 'linear'
        
        # Residual Independence Check (using best model)
        # 1. Linear Correlation (Pearson) - good for linear leakage
        indep_score_linear, _ = pearsonr(np.abs(predictor), np.abs(best_residuals))
        
        # 2. HSIC (Hilbert-Schmidt Independence Criterion) - GOLD STANDARD for non-linear independence
        # More robust than MI for complex non-linear patterns (e.g., tanh+cos in ANM)
        hsic_score = self._hsic_independence(predictor, best_residuals, kernel_width='auto')
        
        # 3. Mutual Information (MI) - Backup measure
        # Normalize residuals to [0, 1] to avoid scale issues
        resid_normalized = (best_residuals - best_residuals.min()) / (best_residuals.max() - best_residuals.min() + 1e-8)
        mi_score = mutual_info_regression(predictor.reshape(-1, 1), resid_normalized, discrete_features=False, random_state=42)[0]
        
        # Heteroscedasticity Check (Binned Variance)
        hetero_score = self._check_heteroscedasticity(predictor, best_residuals)
        
        # Residual Normality (LiNGAM Key: Non-Gaussianity)
        res_skew = skew(best_residuals)
        res_kurt = kurtosis(best_residuals)
        
        return {
            'resid_indep_score': indep_score_linear, # Keep for backward compatibility
            'resid_hsic_score': hsic_score,         # NEW GOLD STANDARD: HSIC for non-linear independence
            'resid_mi_score': mi_score,             # Backup: MI (normalized)
            'resid_hetero_score': hetero_score,
            'resid_skew': res_skew,
            'resid_kurt': res_kurt,
            'r2': best_r2,
            'r2_linear': r2_linear,
            'r2_poly2': r2_poly2,
            'r2_poly3': r2_poly3,
            'best_model': best_model_type,
            'nonlinearity_detected': (best_model_type != 'linear')
        }

    def _hsic_independence(self, X, Y, kernel_width=1.0):
        """
        Hilbert-Schmidt Independence Criterion (HSIC).
        A more robust measure of independence for non-linear relationships.
        Lower HSIC = More Independent. Returns normalized score in [0, 1] range.
        
        Args:
            X, Y: 1D arrays to test for independence
            kernel_width: RBF kernel bandwidth (default=1.0, auto-tuned if median heuristic)
        
        Returns:
            hsic_score: Normalized HSIC value (lower = more independent)
        """
        n = len(X)
        if n < 10:
            return 0.0  # Too few samples
        
        # Reshape to column vectors
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        
        # Use median heuristic for kernel width if not specified
        if kernel_width == 'auto':
            median_x = np.median(pdist(X))
            median_y = np.median(pdist(Y))
            kernel_width_x = median_x if median_x > 0 else 1.0
            kernel_width_y = median_y if median_y > 0 else 1.0
        else:
            kernel_width_x = kernel_width_y = kernel_width
        
        # Compute RBF kernels
        def rbf_kernel(x, width):
            pairwise_sq_dists = squareform(pdist(x, 'sqeuclidean'))
            return np.exp(-pairwise_sq_dists / (2 * width ** 2))
        
        K = rbf_kernel(X, kernel_width_x)
        L = rbf_kernel(Y, kernel_width_y)
        
        # Center the kernels
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ K @ H
        L_c = H @ L @ H
        
        # Compute HSIC
        hsic = np.trace(K_c @ L_c) / (n ** 2)
        
        # Normalize to [0, 1] range (approximation)
        # HSIC can be arbitrarily large, so we use a soft normalization
        hsic_normalized = min(hsic * 10, 1.0)  # Scale factor empirically chosen
        
        return hsic_normalized

    def _check_heteroscedasticity(self, X, resid, n_bins=5):
        """
        Check for heteroscedasticity by comparing variance of residuals across bins of X.
        Returns: Coefficient of Variation of variances (std/mean). Higher = More Heteroscedastic.
        """
        try:
            # Sort by X
            sort_idx = np.argsort(X.flatten())
            resid_sorted = resid[sort_idx]
            
            # Split into bins
            chunks = np.array_split(resid_sorted, n_bins)
            
            # Calculate variance for each bin
            vars = [np.var(c) for c in chunks if len(c) > 1]
            
            if not vars or np.mean(vars) == 0:
                return 0.0
                
            # Coefficient of variation of the variances
            return np.std(vars) / np.mean(vars)
        except:
            return 0.0

    def generate_narrative(self, stats):
        """
        Convert the statistical dictionary into a natural language story.
        """
        lines = []
        
        # Section 1: Individual Behaviors
        lines.append("### Statistical Profile")
        lines.append(f"We are analyzing a dataset with {stats['len']} samples.")
        
        if stats.get('is_discrete', False):
            # Discrete Profile
            lines.append(f"Variable A: Discrete with {stats['x_unique']} unique values. Entropy={stats['x_entropy']:.2f}")
            lines.append(f"Variable B: Discrete with {stats['y_unique']} unique values. Entropy={stats['y_entropy']:.2f}")
            
            # Marginal Entropy Asymmetry
            if stats['x_entropy'] < stats['y_entropy'] - 0.1:
                lines.append("Marginal Entropy: Variable A is significantly 'simpler' (lower entropy) than B. In many causal structures, entropy increases from cause to effect due to noise accumulation.")
            elif stats['y_entropy'] < stats['x_entropy'] - 0.1:
                lines.append("Marginal Entropy: Variable B is significantly 'simpler' (lower entropy) than A. In many causal structures, entropy increases from cause to effect due to noise accumulation.")
            else:
                lines.append("Marginal Entropy: Both variables have similar entropy levels.")
                
            lines.append(f"Variables appear to be categorical or ordinal.")
        else:
            # Continuous Profile
            def interpret_skew(v):
                if abs(v) > 1.0: return "Highly Skewed (e.g., Exponential-like)"
                if abs(v) > 0.5: return "Moderately Skewed"
                return "Symmetric (Gaussian-like)"
                
            lines.append(f"Variable A: Skewness={stats['x_skew']:.2f} ({interpret_skew(stats['x_skew'])}), Kurtosis={stats['x_kurt']:.2f}.")
            lines.append(f"Variable B: Skewness={stats['y_skew']:.2f} ({interpret_skew(stats['y_skew'])}), Kurtosis={stats['y_kurt']:.2f}.")
            
            # Correlation
            corr_strength = "weak"
            if abs(stats['pearson_corr']) > 0.7: corr_strength = "strong"
            elif abs(stats['pearson_corr']) > 0.3: corr_strength = "moderate"
            lines.append(f"Correlation: {corr_strength} (Pearson r={stats['pearson_corr']:.2f}).")
        
        lines.append("\n### Causal Mechanism Analysis")
        lines.append("We tested two competing causal hypotheses by fitting models in both directions.")
        
        if stats.get('is_discrete', False):
            # === Discrete Narrative ===
            def describe_discrete(direction_name, metrics):
                desc = [f"\n**Hypothesis: {direction_name}**"]
                desc.append(f"- Model Type: {metrics['model_type']}")
                desc.append(f"- Classification Accuracy: {metrics['accuracy']:.3f}")
                desc.append(f"- Conditional Entropy: {metrics['conditional_entropy']:.3f} (Lower = Better predictability)")
                desc.append(f"- Mutual Information (X,Y): {metrics['mutual_information']:.3f} (Higher = Stronger association)")
                
                # Error Independence
                p_val = metrics['error_independence_p']
                if p_val > 0.05:
                    desc.append(f"- Error Independence: **High** (p-value={p_val:.3f}). Prediction errors are independent of the input, suggesting a correct causal model.")
                else:
                    desc.append(f"- Error Independence: **Low** (p-value={p_val:.3f}). Prediction errors depend on the input, suggesting model misspecification.")
                return "\n".join(desc)

            lines.append(describe_discrete("A -> B", stats['dir_ab']))
            lines.append(describe_discrete("B -> A", stats['dir_ba']))
            
            lines.append("\n### Comparative Analysis (Objective)")
            acc_ab = stats['dir_ab']['accuracy']
            acc_ba = stats['dir_ba']['accuracy']
            ent_ab = stats['dir_ab']['conditional_entropy']
            ent_ba = stats['dir_ba']['conditional_entropy']
            p_ab = stats['dir_ab']['error_independence_p']
            p_ba = stats['dir_ba']['error_independence_p']
            mi_ab = stats['dir_ab']['mutual_information']
            mi_ba = stats['dir_ba']['mutual_information']
            
            # 1. Predictive Determinism (Conditional Entropy - Lower is better)
            ent_diff = ent_ba - ent_ab  # Positive means A->B has lower entropy (better)
            ent_rel_diff = abs(ent_diff) / (min(ent_ab, ent_ba) + 1e-9)
            
            lines.append(f"- **Predictive Determinism (Conditional Entropy)**: A->B: {ent_ab:.4f} vs B->A: {ent_ba:.4f}.")
            if ent_rel_diff < 0.05:  # < 5% difference
                lines.append(f"  -> The conditional entropies are nearly identical (relative difference < 5%). Both directions have similar predictive power.")
            elif ent_rel_diff < 0.15:  # 5-15% difference
                better_dir = "A->B" if ent_diff > 0 else "B->A"
                lines.append(f"  -> Direction {better_dir} shows **slightly lower** conditional entropy (relative difference ~{ent_rel_diff*100:.0f}%), indicating moderately better predictability.")
            else:  # > 15% difference
                better_dir = "A->B" if ent_diff > 0 else "B->A"
                lines.append(f"  -> Direction {better_dir} has **notably lower** conditional entropy (relative difference ~{ent_rel_diff*100:.0f}%), meaning it reduces uncertainty more effectively.")
            
            # 2. Error Independence (Higher p-value is better)
            lines.append(f"\n- **Error Independence Test**: A->B p-value: {p_ab:.4f} vs B->A p-value: {p_ba:.4f}.")
            lines.append(f"  -> Interpretation: p-value > 0.05 suggests errors are independent of input (good model fit).")
            
            if p_ab > 0.05 and p_ba > 0.05:
                lines.append(f"  -> Both directions pass the independence test. This metric alone cannot distinguish the causal direction.")
            elif p_ab < 0.05 and p_ba < 0.05:
                lines.append(f"  -> Both directions fail the independence test. The relationship may be more complex or confounded.")
            else:
                pass_dir = "A->B" if p_ab > 0.05 else "B->A"
                fail_dir = "B->A" if p_ab > 0.05 else "A->B"
                lines.append(f"  -> Direction {pass_dir} passes (p={p_ab if p_ab > 0.05 else p_ba:.4f}), while {fail_dir} fails (p={p_ba if p_ab > 0.05 else p_ab:.4f}). This asymmetry is a notable signal.")
            
            # 3. Mutual Information Symmetry Check
            mi_avg = (mi_ab + mi_ba) / 2
            lines.append(f"\n- **Mutual Information**: A->B: {mi_ab:.4f}, B->A: {mi_ba:.4f} (Average: {mi_avg:.4f}).")
            lines.append(f"  -> Note: MI(X,Y) should theoretically be symmetric. Asymmetry here reflects directional prediction strength in the discrete context.")
            
        else:
            # === Continuous Narrative (Existing Logic) ===
            def describe_model(direction_name, metrics):
                desc = [f"\n**Hypothesis: {direction_name}**"]
                
                # Model fit quality
                desc.append(f"- Best Model: {metrics['best_model']} (R²={metrics['r2']:.3f})")
                
                # Non-linearity signal
                if metrics['nonlinearity_detected']:
                    r2_gain = metrics['r2'] - metrics['r2_linear']
                    desc.append(f"- Non-linearity Detected: R² improved by {r2_gain:.3f} with polynomial fit (Linear R²={metrics['r2_linear']:.3f})")
                else:
                    desc.append(f"- Linear Model Sufficient: Polynomial fit did not significantly improve R² (Linear={metrics['r2_linear']:.3f})")
                
                # Independence (Using HSIC as primary metric - more robust for non-linear patterns)
                # HSIC Score: 0 = Independent, Higher = Dependent
                hsic = metrics.get('resid_hsic_score', metrics.get('resid_mi_score', 0))  # Fallback to MI if HSIC not available
                hetero = metrics['resid_hetero_score']
                
                if hsic < 0.1:
                    desc.append(f"- Residual Independence: **High** (HSIC: {hsic:.3f}, Heterogeneity: {hetero:.2f}). The residuals are effectively independent of the predictor.")
                elif hsic < 0.3:
                    desc.append(f"- Residual Independence: **Moderate** (HSIC: {hsic:.3f}, Heterogeneity: {hetero:.2f}). Some dependency detected.")
                else:
                    desc.append(f"- Residual Independence: **Low** (HSIC: {hsic:.3f}, Heterogeneity: {hetero:.2f}). Strong dependency detected between predictor and residuals (model failed to capture structure).")
                
                # Non-Gaussianity (Entropy proxy)
                kurt = metrics['resid_kurt']
                if abs(kurt) > 0.5:
                     desc.append(f"- Residual Distribution: Clearly Non-Gaussian (Kurtosis: {kurt:.2f}). This is a good sign for a causal mechanism if the noise is non-Gaussian.")
                else:
                     desc.append(f"- Residual Distribution: Gaussian-like (Kurtosis: {kurt:.2f}).")
                
                return "\n".join(desc)

            lines.append(describe_model("A -> B", stats['dir_ab']))
            lines.append(describe_model("B -> A", stats['dir_ba']))

            # Comparative Summary
            lines.append("\n### Comparative Analysis (Objective)")
            # Use HSIC as primary metric (fallback to MI for backward compatibility)
            score_ab = stats['dir_ab'].get('resid_hsic_score', stats['dir_ab']['resid_mi_score'])
            score_ba = stats['dir_ba'].get('resid_hsic_score', stats['dir_ba']['resid_mi_score'])
            r2_ab = stats['dir_ab']['r2']
            r2_ba = stats['dir_ba']['r2']
            
            diff_score = score_ba - score_ab  # Positive means A->B has lower score (better independence)
            rel_diff = abs(diff_score) / (min(score_ab, score_ba) + 1e-9)  # Relative difference
            
            # 1. Model Fit Comparison (Goodness of Fit)
            r2_diff = r2_ab - r2_ba
            r2_rel_diff = abs(r2_diff) / (max(r2_ab, r2_ba) + 1e-9)
            
            lines.append(f"- **Model Fit (R²)**: A->B: {r2_ab:.4f} vs B->A: {r2_ba:.4f}.")
            if r2_rel_diff < 0.02:  # < 2% difference
                lines.append(f"  -> Both directions explain the data equally well (relative difference < 2%).")
            else:
                better_dir = "A->B" if r2_diff > 0 else "B->A"
                worse_dir = "B->A" if r2_diff > 0 else "A->B"
                lines.append(f"  -> Direction {better_dir} achieves better fit (R² is {abs(r2_diff):.4f} or ~{r2_rel_diff*100:.0f}% higher than {worse_dir}).")
            
            # 2. Mechanism Complexity Analysis
            nonlin_ab = stats['dir_ab']['nonlinearity_detected']
            nonlin_ba = stats['dir_ba']['nonlinearity_detected']
            
            lines.append(f"\n- **Mechanism Complexity**: A->B: {stats['dir_ab']['best_model']}, B->A: {stats['dir_ba']['best_model']}.")
            if nonlin_ab and not nonlin_ba:
                lines.append(f"  -> A->B requires a non-linear model (R²={r2_ab:.3f}), while B->A is adequately modeled as linear (R²={r2_ba:.3f}).")
                lines.append(f"  -> Note: In ANM theory, if the non-linear model achieves much better independence, complexity is justified.")
            elif nonlin_ba and not nonlin_ab:
                lines.append(f"  -> B->A requires a non-linear model (R²={r2_ba:.3f}), while A->B is adequately modeled as linear (R²={r2_ab:.3f}).")
                lines.append(f"  -> Note: In ANM theory, if the non-linear model achieves much better independence, complexity is justified.")
            else:
                lines.append(f"  -> Both directions imply similar functional complexity.")
            
            # 3. Residual Independence Comparison (THE KEY METRIC)
            metric_name = "HSIC" if 'resid_hsic_score' in stats['dir_ab'] else "MI"
            lines.append(f"\n- **Residual Independence ({metric_name})**: A->B: {score_ab:.4f} vs B->A: {score_ba:.4f}.")
            lines.append(f"  -> Lower score = Better independence (residuals are decoupled from predictor).")
            
            if rel_diff < 0.1:  # < 10% relative difference
                lines.append(f"  -> The independence scores are very close (relative difference < 10%). Distinguishing direction based solely on this metric is challenging.")
            elif rel_diff < 0.5:  # 10-50% difference
                better_dir = "A->B" if diff_score > 0 else "B->A"
                lines.append(f"  -> Direction {better_dir} shows **moderately better** residual independence (relative improvement ~{rel_diff*100:.0f}%).")
            else:  # > 50% difference
                better_dir = "A->B" if diff_score > 0 else "B->A"
                lines.append(f"  -> Direction {better_dir} demonstrates **substantially better** residual independence (relative improvement >{rel_diff*100:.0f}%). This is a strong signal.")
            
            # 4. Heteroscedasticity Signal
            hetero_ab = stats['dir_ab']['resid_hetero_score']
            hetero_ba = stats['dir_ba']['resid_hetero_score']
            lines.append(f"\n- **Heteroscedasticity**: A->B: {hetero_ab:.3f}, B->A: {hetero_ba:.3f}.")
            lines.append(f"  -> Higher value indicates variance of residuals changes with predictor (often a sign of wrong direction in ANM).")

        return "\n".join(lines)
