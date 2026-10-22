import argparse
import csv
import html
import re
from pathlib import Path

KEYWORDS = {
    'excellent': 'Expected positive',
    'great': 'Expected positive',
    'wonderful': 'Expected positive',
    'awful': 'Expected negative',
    'terrible': 'Expected negative',
    'boring': 'Expected negative',
}


def tokenize(review):
    text = re.sub(r'<[^>]+>', ' ', html.unescape(review))
    return set(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))


def count_reviews(csv_path):
    total = positive = 0
    counts = {word: {'all': 0, 'positive': 0} for word in KEYWORDS}
    with open(csv_path, encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not {'review', 'sentiment'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain review and sentiment columns.')
        for line, row in enumerate(reader, start=2):
            label = row['sentiment'].strip().lower()
            if label not in {'positive', 'negative'}:
                raise ValueError(f'Unexpected sentiment on CSV line {line}: {label!r}')
            total += 1
            positive += label == 'positive'
            words = tokenize(row['review'])
            for word in KEYWORDS:
                if word in words:
                    counts[word]['all'] += 1
                    counts[word]['positive'] += label == 'positive'
    if total == 0 or positive == 0:
        raise ValueError('Need a nonempty dataset with positive reviews.')
    return total, positive, counts


def calculate_probabilities(total, positive, counts):
    results = []
    for word, count in counts.items():
        prior = positive / total
        likelihood = count['positive'] / positive
        marginal = count['all'] / total
        # Bayes: P(Positive | word) = P(word | Positive) P(Positive) / P(word)
        posterior = likelihood * prior / marginal if marginal else None
        results.append({
            'keyword': word, 'selection': KEYWORDS[word],
            'reviews_with_keyword': count['all'],
            'positive_reviews_with_keyword': count['positive'],
            'prior': prior, 'likelihood': likelihood,
            'marginal': marginal, 'posterior': posterior,
        })
    return results


def markdown_table(results):
    lines = [
        '| Keyword | Reviews containing word | Positive containing word | P(Positive) | P(word given Positive) | P(word) | P(Positive given word) |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in results:
        probabilities = [f"{row[key]:.6f}" if row[key] is not None else 'undefined'
                         for key in ('prior', 'likelihood', 'marginal', 'posterior')]
        lines.append('| ' + ' | '.join([
            row['keyword'], str(row['reviews_with_keyword']),
            str(row['positive_reviews_with_keyword']), *probabilities]) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Path to IMDB Dataset.csv')
    parser.add_argument('--output', type=Path, default=Path('keyword_probabilities.csv'))
    args = parser.parse_args()
    total, positive, counts = count_reviews(args.dataset)
    results = calculate_probabilities(total, positive, counts)
    print(f'Reviews: {total:,}; positive: {positive:,}; negative: {total-positive:,}\n')
    print(markdown_table(results))
    with args.output.open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
