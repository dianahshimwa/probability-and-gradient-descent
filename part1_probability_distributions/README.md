# Part 1: EM on Galton's Heights (Fathers vs. Children)

A from-scratch Expectation-Maximization fit of a 2-component Gaussian mixture to 1,139 unlabeled heights:
934 children and 205 fathers from Galton's 1886 family data (the [Kaggle dataset](https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data), same file as `HistData::GaltonFamilies`).

## Files
| File | What it is |
|---|---|
| `em_gmm.py` | The algorithm: `init_params`, `gaussian_pdf`, `e_step`, `m_step`, `log_likelihood`, `fit_em`, `classify`. NumPy only, no ML libraries. |
| `plots.py` | All figures, saved to `images/` |
| `demo.py` | Live test script for the coach's height |
| `part1_em_algorithm.ipynb` | Full walkthrough with outputs and figures (runs locally or in Colab) |
| `data/GaltonFamilies.csv` | Dataset |

## Run
From inside `part1_probability_distributions/`:
```bash
pip install -r requirements.txt
python demo.py 70              # inches
python demo.py 178 --unit cm   # centimetres
python demo.py 70 --table      # also prints the tracking table
```
Open in Colab: go to **File → Open notebook → GitHub**, paste the repo URL, and pick `part1_probability_distributions/part1_em_algorithm.ipynb`. The first cell clones the repo, so the imports work.

## Tracking table
| Iteration | μ1 (Children) | μ2 (Fathers) | σ1² | σ2² | π1 | π2 | Log-Likelihood |
|---|---|---|---|---|---|---|---|
| 0 (Initialization) | 64.5000 | 70.0000 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | -3119.8222 |
| 1 | 65.2205 | 69.2054 | 9.0603 | 8.4468 | 0.5011 | 0.4989 | -3057.9135 |
| 2 | 65.1609 | 69.2600 | 8.9420 | 8.1040 | 0.5005 | 0.4995 | -3056.9719 |
| 319 (converged) | 64.1772 | 69.5186 | 5.3024 | 6.0410 | 0.4325 | 0.5675 | -3046.9895 |

Initialization: means at the 25th/75th percentile, both variances equal to the global variance, π = (0.5, 0.5). Stopping rule: |Δ log-likelihood| < 10⁻⁶.

## Should we split at the global mean? No.
| | Mean-split pile 1 | True children | Mean-split pile 2 | True fathers |
|---|---|---|---|---|
| mean | 64.26 | 66.75 | 70.21 | 69.32 |
| std dev | 2.04 | 3.58 | 1.89 | 2.64 |
| count | 575 | 934 | 564 | 205 |

1. **Wrong cut.** The global mean (67.21 in) is pulled toward the larger group (4.5× more children). It is not where the populations meet.
2. **Overlap means misclassification.** Tall sons and short fathers end up on the wrong side of the cut.
3. **Truncation bias.** Each pile's mean is pushed away from the cut and its spread is roughly halved. The piles come out 50/50 when the truth is 82/18.
4. **No uncertainty.** A 67.0 in person counts 100% as "child". EM's soft responsibilities let each point near the boundary contribute partly to both components.

## Finding
EM converges to the highest-likelihood solution (it matches from both initialisations, and log-likelihood never decreases). But the children are themselves bimodal: daughters average 64.1 in and sons 69.2 in. So EM separates **daughters** from **sons + fathers**, because sex drives height more than generation does. "P(Child)" therefore means "P(shorter population)". Fixes: a 3-component mixture, or semi-supervised EM.
