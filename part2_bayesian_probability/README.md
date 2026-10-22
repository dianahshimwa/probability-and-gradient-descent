# Part 2: Bayesian Probability with IMDb Reviews

## Research question
Given that a review contains a selected keyword, what is the probability that its sentiment label is positive?

We compute **P(Positive | keyword) only**, for every keyword. Negative-indicator words are included in that same question; we do not calculate P(Negative | keyword).

## Dataset and method
Source: [IMDb Dataset of 50K Movie Reviews, Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
The downloaded CSV contains 50,000 reviews: 25,000 positive and 25,000 negative. All rows are used, including repeated reviews if present; no deduplication or sampling is applied.

Chosen beforehand: excellent, great, wonderful (expected positive); awful, terrible, boring (expected negative). These adjectives express praise, enjoyment, dislike, or dissatisfaction.

The implementation uses only the Python standard library: no pandas, NumPy, or machine learning libraries. It decodes HTML entities, removes HTML tags, converts text to lowercase, and extracts whole-word tokens. Each review contributes at most one count per keyword. Thus “great great” counts once, and “greater” does not match “great.” Inflections are not merged. A review may count for more than one keyword.

## Bayes’ theorem
Let A mean “the review is positive” and K mean “the review contains the keyword.”

P(A | K) = P(K | A) × P(A) / P(K)

- Prior P(A) = number of positive reviews / total reviews.
- Likelihood P(K | A) = positive reviews containing keyword / positive reviews.
- Marginal P(K) = all reviews containing keyword / total reviews.
- Posterior P(A | K) = likelihood × prior / marginal.

The likelihood and posterior answer different questions: how common the word is among positive reviews versus how common positive labels are among reviews with the word.

## Computed results
Probabilities below are rounded to six decimal places. The CSV retains full floating-point values.

| Keyword | Reviews containing word | Positive containing word | P(Positive) | P(word given Positive) | P(word) | P(Positive given word) |
|---|---:|---:|---:|---:|---:|---:|
| excellent | 3552 | 2868 | 0.500000 | 0.114720 | 0.071040 | 0.807432 |
| great | 12576 | 8477 | 0.500000 | 0.339080 | 0.251520 | 0.674062 |
| wonderful | 2778 | 2256 | 0.500000 | 0.090240 | 0.055560 | 0.812095 |
| awful | 2882 | 283 | 0.500000 | 0.011320 | 0.057640 | 0.098196 |
| terrible | 2699 | 382 | 0.500000 | 0.015280 | 0.053980 | 0.141534 |
| boring | 3052 | 592 | 0.500000 | 0.023680 | 0.061040 | 0.193971 |

## Worked example: excellent
- Prior = 25,000 / 50,000 = 0.5.
- Likelihood = 2,868 / 25,000 = 0.11472.
- Marginal = 3,552 / 50,000 = 0.07104.
- Posterior = (0.11472 × 0.5) / 0.07104 = 0.8074324324, or approximately **80.74%**.

As a cross-check, 2,868 / 3,552 gives the same posterior. The implementation explicitly uses Bayes’ formula.

## Interpretation and limitations
The starting probability is 50%. All three expected positive words raise it; all three expected negative words lower it. “Wonderful” has the highest positive posterior among our six choices (81.21%), and “awful” the lowest (9.82%). This supports our keyword expectations in this dataset.

These are empirical conditional probabilities from the complete dataset, not a measured accuracy score on unseen reviews. We do not train or evaluate a full sentiment classifier. Negation (“not wonderful”), sarcasm, and descriptions of an actor or character can break the relationship between one word and the review’s overall label. Dataset balance also affects the prior, so these probabilities need not apply to another review population. We do not combine keyword probabilities or assume keyword independence. No smoothing is used; an unseen keyword has an undefined posterior, which the code reports explicitly.

## Run for your presentation
1. Download and unzip the dataset from the source above.
2. Put `IMDB Dataset.csv` beside `part2_bayes.py`, or supply its actual path.
3. Run:

```sh
python3 part2_bayes.py "IMDB Dataset.csv" --output keyword_probabilities.csv
```

The script prints dataset counts and the complete probability table, then writes the results CSV. The notebook provides the same analysis with explanatory cells. Set `DATASET` in its first code cell if needed. No third-party packages are required by the analysis script; a Jupyter environment is needed to open and run the notebook.

Suggested speaking notes:
“Our prior probability is one half because the dataset has equal numbers of positive and negative reviews. Bayes’ theorem updates that starting probability using the presence of a keyword. Of 3,552 reviews containing excellent, 2,868 are positive, giving a posterior of about 80.74 percent. We ask the same positive-sentiment question for awful, where the result is about 9.82 percent. This shows that words provide useful evidence, although context such as negation can change their meaning.”
