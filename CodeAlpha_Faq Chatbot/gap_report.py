"""
Summarizes data/unanswered.csv so you know exactly which FAQs to add next.
Run after the bot has been used for a while: python gap_report.py
"""

import csv
import os
from collections import Counter

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "unanswered.csv")


def main():
    if not os.path.exists(LOG_PATH):
        print("No unanswered questions logged yet — nothing to report.")
        return

    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Log file is empty — nothing to report.")
        return

    counts = Counter(row["question"].strip().lower() for row in rows)
    print(f"{len(rows)} unanswered questions logged, {len(counts)} unique.\n")
    print("Most frequent gaps (consider adding these to faqs.csv):\n")
    for question, n in counts.most_common(15):
        print(f"  {n:>3}x  {question}")


if __name__ == "__main__":
    main()
