from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

CLASSES = ["no", "neutral", "yes"]

HEADERS = [
    "buyer_income_lpa",
    "house_price_lakh",
    "loan_eligibility",
    "credit_score",
    "down_payment_percent",
    "existing_emi_lpa",
    "employment_years",
    "dependents",
    "property_location_score",
    "employment_type",
    "can_buy",
]

SCENARIOS = {
    "entry_level": {
        "weight": 0.26,
        "income_tri": (2.2, 16.0, 7.5),
        "price_ratio_tri": (3.8, 17.5, 8.8),
        "credit_mean": 660,
        "credit_std": 85,
        "down_payment_tri": (2.0, 25.0, 11.0),
        "emi_ratio_tri": (0.05, 0.58, 0.28),
        "emp_years_tri": (0.0, 15.0, 4.0),
        "loan_yes_base": 0.52,
        "employment_weights": [0.53, 0.32, 0.15],
    },
    "mid_tier": {
        "weight": 0.31,
        "income_tri": (8.0, 38.0, 20.0),
        "price_ratio_tri": (2.8, 12.0, 5.6),
        "credit_mean": 710,
        "credit_std": 65,
        "down_payment_tri": (7.0, 42.0, 18.0),
        "emi_ratio_tri": (0.02, 0.45, 0.19),
        "emp_years_tri": (1.0, 25.0, 8.5),
        "loan_yes_base": 0.73,
        "employment_weights": [0.59, 0.21, 0.20],
    },
    "premium": {
        "weight": 0.18,
        "income_tri": (24.0, 95.0, 45.0),
        "price_ratio_tri": (2.0, 8.0, 4.1),
        "credit_mean": 780,
        "credit_std": 45,
        "down_payment_tri": (15.0, 65.0, 30.0),
        "emi_ratio_tri": (0.0, 0.35, 0.12),
        "emp_years_tri": (2.0, 34.0, 12.0),
        "loan_yes_base": 0.90,
        "employment_weights": [0.65, 0.14, 0.21],
    },
    "stressed": {
        "weight": 0.15,
        "income_tri": (2.0, 30.0, 10.0),
        "price_ratio_tri": (4.0, 20.0, 10.5),
        "credit_mean": 605,
        "credit_std": 90,
        "down_payment_tri": (1.0, 20.0, 7.0),
        "emi_ratio_tri": (0.15, 0.72, 0.42),
        "emp_years_tri": (0.0, 20.0, 3.0),
        "loan_yes_base": 0.39,
        "employment_weights": [0.38, 0.48, 0.14],
    },
    "investor": {
        "weight": 0.10,
        "income_tri": (12.0, 80.0, 30.0),
        "price_ratio_tri": (2.8, 15.0, 7.2),
        "credit_mean": 735,
        "credit_std": 70,
        "down_payment_tri": (10.0, 60.0, 24.0),
        "emi_ratio_tri": (0.06, 0.56, 0.27),
        "emp_years_tri": (1.0, 30.0, 9.0),
        "loan_yes_base": 0.70,
        "employment_weights": [0.34, 0.22, 0.44],
    },
}

EMPLOYMENT_TYPES = ["salaried", "self_employed", "business"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def split_counts(total: int) -> dict[str, int]:
    base = total // len(CLASSES)
    remainder = total % len(CLASSES)
    counts = {cls_name: base for cls_name in CLASSES}
    for i in range(remainder):
        counts[CLASSES[i]] += 1
    return counts


def score_target(
    income: float,
    price: float,
    loan_eligibility: str,
    credit_score: int,
    down_payment_percent: float,
    existing_emi_lpa: float,
    employment_years: int,
    dependents: int,
    property_location_score: int,
    employment_type: str,
    rng: random.Random,
) -> str:
    score = 0.0

    if loan_eligibility == "yes":
        score += 2.0
    else:
        score -= 2.2

    if credit_score >= 810:
        score += 3.2
    elif credit_score >= 750:
        score += 2.2
    elif credit_score >= 680:
        score += 1.1
    elif credit_score < 610:
        score -= 2.4

    price_to_income = price / max(income, 0.1)
    emi_ratio = existing_emi_lpa / max(income, 0.1)

    if price_to_income <= 3.8:
        score += 3.3
    elif price_to_income <= 5.8:
        score += 2.2
    elif price_to_income <= 8.2:
        score += 1.0
    elif price_to_income >= 17.0:
        score -= 5.2
    elif price_to_income >= 12.5:
        score -= 3.4

    if emi_ratio > 0.52:
        score -= 3.4
    elif emi_ratio > 0.38:
        score -= 2.2
    elif emi_ratio < 0.08:
        score += 0.7

    if employment_years >= 12:
        score += 2.0
    elif employment_years >= 5:
        score += 1.0
    elif employment_years <= 1:
        score -= 1.1

    if employment_type == "salaried":
        score += 1.0
    elif employment_type == "business":
        score += 0.6

    if down_payment_percent >= 35:
        score += 2.5
    elif down_payment_percent >= 22:
        score += 1.3
    elif down_payment_percent < 8:
        score -= 2.3

    if dependents >= 5:
        score -= 1.2
    elif dependents == 0:
        score += 0.2

    if property_location_score >= 8:
        score += 0.6
    elif property_location_score <= 3:
        score -= 0.5

    if loan_eligibility == "no" and price > income * 5.5:
        score -= 2.5

    # Add mild uncertainty near boundaries to increase realistic label variation.
    score += rng.uniform(-0.55, 0.55)
    yes_threshold = 5.0 + rng.uniform(-0.45, 0.45)
    neutral_threshold = 1.5 + rng.uniform(-0.35, 0.35)

    if score >= yes_threshold:
        return "yes"
    if score >= neutral_threshold:
        return "neutral"
    return "no"


def make_random_row(rng: random.Random) -> tuple:
    scenario_name = rng.choices(
        population=list(SCENARIOS.keys()),
        weights=[SCENARIOS[name]["weight"] for name in SCENARIOS],
        k=1,
    )[0]
    sc = SCENARIOS[scenario_name]

    income = rng.triangular(*sc["income_tri"])
    price_ratio = rng.triangular(*sc["price_ratio_tri"])
    price = income * price_ratio + rng.uniform(-18.0, 18.0)

    # Add occasional high-end / stressed outliers to broaden variance.
    if rng.random() < 0.08:
        income *= rng.uniform(0.7, 1.6)
        price *= rng.uniform(0.8, 1.8)

    credit_score = int(round(rng.gauss(sc["credit_mean"], sc["credit_std"])))
    if rng.random() < 0.12:
        credit_score += int(rng.gauss(0, 45))
    credit_score = int(clamp(credit_score, 470, 900))

    down_payment_percent = rng.triangular(*sc["down_payment_tri"])
    emi_ratio = rng.triangular(*sc["emi_ratio_tri"])
    existing_emi_lpa = income * emi_ratio + rng.uniform(-0.75, 0.75)
    employment_years = int(round(rng.triangular(*sc["emp_years_tri"])))
    dependents = rng.choices([0, 1, 2, 3, 4, 5, 6], weights=[13, 20, 24, 18, 13, 8, 4], k=1)[0]
    property_location_score = rng.choices(range(1, 11), weights=[5, 6, 8, 10, 12, 14, 14, 13, 10, 8], k=1)[0]
    employment_type = rng.choices(EMPLOYMENT_TYPES, weights=sc["employment_weights"], k=1)[0]

    income = round(clamp(income, 1.8, 145.0), 2)
    price = round(clamp(price, 20.0, 1200.0), 2)
    down_payment_percent = round(clamp(down_payment_percent, 1.0, 70.0), 2)
    existing_emi_lpa = round(clamp(existing_emi_lpa, 0.0, max(1.0, income * 0.82)), 2)
    employment_years = int(clamp(employment_years, 0, 35))

    loan_prob = sc["loan_yes_base"]
    loan_prob += 0.16 if credit_score >= 760 else 0.0
    loan_prob -= 0.18 if credit_score < 600 else 0.0
    loan_prob -= 0.22 if (existing_emi_lpa / max(income, 0.1)) > 0.45 else 0.0
    loan_prob -= 0.10 if down_payment_percent < 8 else 0.0
    loan_prob += 0.07 if income > 35 else 0.0
    loan_prob = clamp(loan_prob, 0.05, 0.97)
    loan_eligibility = "yes" if rng.random() < loan_prob else "no"

    can_buy = score_target(
        income,
        price,
        loan_eligibility,
        credit_score,
        down_payment_percent,
        existing_emi_lpa,
        employment_years,
        dependents,
        property_location_score,
        employment_type,
        rng,
    )

    return (
        income,
        price,
        loan_eligibility,
        credit_score,
        down_payment_percent,
        existing_emi_lpa,
        employment_years,
        dependents,
        property_location_score,
        employment_type,
        can_buy,
    )


def generate_dataset(out_dir: Path, train_size: int, cv_size: int, test_size: int, seed: int) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_plan = {
        "train": split_counts(train_size),
        "cv": split_counts(cv_size),
        "test": split_counts(test_size),
    }

    needed_total = {cls_name: 0 for cls_name in CLASSES}
    for counts in split_plan.values():
        for cls_name, cnt in counts.items():
            needed_total[cls_name] += cnt

    pools: dict[str, list[tuple]] = {cls_name: [] for cls_name in CLASSES}
    seen = set()
    max_attempts = 12_000_000
    attempts = 0

    while any(len(pools[cls_name]) < needed_total[cls_name] for cls_name in CLASSES):
        row = make_random_row(rng)
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Could not generate balanced pools within attempt limit.")

        row_key = tuple(row[:-1])  # uniqueness by features, not label
        if row_key in seen:
            continue
        seen.add(row_key)

        target = row[-1]
        if len(pools[target]) < needed_total[target]:
            pools[target].append(row)

    for cls_name in CLASSES:
        rng.shuffle(pools[cls_name])

    split_rows = {"train": [], "cv": [], "test": []}
    cursor = {cls_name: 0 for cls_name in CLASSES}

    for split_name in ["train", "cv", "test"]:
        for cls_name in CLASSES:
            cnt = split_plan[split_name][cls_name]
            start = cursor[cls_name]
            end = start + cnt
            split_rows[split_name].extend(pools[cls_name][start:end])
            cursor[cls_name] = end
        rng.shuffle(split_rows[split_name])

    for split_name, rows in split_rows.items():
        path = out_dir / f"house_buy_{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
            writer.writerows(rows)

    combined_path = out_dir / "house_buy_full_dataset.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS + ["split"])
        for split_name in ["train", "cv", "test"]:
            for row in split_rows[split_name]:
                writer.writerow(list(row) + [split_name])

    print("Dataset written to:", out_dir)
    for split_name in ["train", "cv", "test"]:
        counts = {cls_name: 0 for cls_name in CLASSES}
        for row in split_rows[split_name]:
            counts[row[-1]] += 1
        print(split_name, len(split_rows[split_name]), counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic house-buying dataset.")
    parser.add_argument("--out-dir", default=r"C:\Users\91960\house_pricing_nn\csv")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--cv-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        out_dir=Path(args.out_dir),
        train_size=args.train_size,
        cv_size=args.cv_size,
        test_size=args.test_size,
        seed=args.seed,
    )
