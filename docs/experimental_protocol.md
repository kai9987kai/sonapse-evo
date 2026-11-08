# Experimental Protocol and Confirmatory Study Design

## 1. Primary Research Hypothesis

An artificial recurrent network can recover damaged functionality more efficiently when a controller uses causal evidence to select and train targeted neural grafts, compared with equally resourced random grafting, gradient-guided rewiring (RigL), ordinary fine-tuning, and capacity-matched adapters.

## 2. Experimental Arms (A0 – A9)

| Arm ID | Architecture | Role |
| :--- | :--- | :--- |
| **A0** | Damaged, frozen network | Negative baseline control |
| **A1** | Ordinary fine-tuning | Standard backpropagation baseline |
| **A2** | Random sparse graft | Extra connection capacity control |
| **A3** | Capacity-matched learned adapter | Extra parameter capacity control |
| **A4** | Gradient-guided rewiring (RigL) | Sparse topology optimization baseline |
| **A5** | Correlation-selected graft | Covariance-based diagnostic baseline |
| **A6** | **SONAPSE-EVO Causal Grafting** | **Primary proposed method** |
| **A7** | Biological motif with causal selection | Biological prior test (Dale's law) |
| **A8** | AETHER-derived graft with causal selection | External donor transfer test |
| **A9** | Oracle lesion-location graft | Upper-bound diagnostic control (non-deployable) |

## 3. Seed Independence & Predeclared Comparisons

- **Exploratory Phase**: $N = 5$ paired training seeds for initial mechanism tuning.
- **Confirmatory Phase**: $N = 30$ independent paired seeds across distinct environment seeds and counter-based random number generators.
- **Predeclared Primary Contrast**: A6 versus A4 (RigL sparse rewiring).
- **Secondary Contrasts**: A6 versus A1 (Fine-Tuning) and A6 versus A3 (Matched Adapter).
- **Statistical Significance**: Two-sided paired $t$-test ($\alpha = 0.05$), non-parametric Wilcoxon signed-rank test, and 95% bootstrap confidence interval of mean difference $\bar{d}$.
- **Effect Size**: Paired Cohen's $d = \frac{\bar{d}}{\sigma_d}$.
