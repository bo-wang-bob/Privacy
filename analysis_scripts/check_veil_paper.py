"""Static AAAI manuscript checks that do not invoke a LaTeX compiler."""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "aaai2027" / "veil.tex"
BIB = ROOT / "paper" / "aaai2027" / "veil.bib"
CHECKLIST = ROOT / "paper" / "aaai2027" / "veil_reproducibility.tex"
DATA = ROOT / "paper" / "aaai2027" / "evidence"
FORBIDDEN_PACKAGES = {
    "authblk",
    "balance",
    "CJK",
    "float",
    "flushend",
    "fullpage",
    "geometry",
    "hyperref",
    "multicol",
    "setspace",
    "stfloats",
    "tabu",
    "titlesec",
    "ulem",
    "wrapfig",
}
FORBIDDEN_COMMANDS = (
    r"\addtolength",
    r"\baselinestretch",
    r"\clearpage",
    r"\newpage",
    r"\pagebreak",
    r"\pagestyle",
    r"\resizebox",
)


def uncommented(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def check_braces(text: str) -> None:
    depth = 0
    for index, character in enumerate(text):
        if character == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif character == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth < 0:
                raise AssertionError(f"Unexpected closing brace at offset {index}")
    if depth:
        raise AssertionError(f"Unbalanced braces: final depth {depth}")


def check_environments(text: str) -> None:
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        action, environment = match.groups()
        if action == "begin":
            stack.append(environment)
        elif not stack or stack.pop() != environment:
            raise AssertionError(f"Mismatched environment near {match.group(0)}")
    if stack:
        raise AssertionError(f"Unclosed environments: {stack}")


def check_citations(text: str, bib: str) -> None:
    available = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for payload in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        cited.update(item.strip() for item in payload.split(","))
    missing = cited - available
    if missing:
        raise AssertionError(f"Missing BibTeX entries: {sorted(missing)}")
    unused = available - cited
    if unused:
        raise AssertionError(f"Unused BibTeX entries: {sorted(unused)}")


def check_figures(text: str) -> None:
    for relative in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        path = PAPER.parent / relative
        if not path.exists():
            raise AssertionError(f"Missing figure: {path}")


def csv_rows(name: str) -> list[dict[str, str]]:
    path = DATA / name
    if not path.exists():
        raise AssertionError(f"Missing validated paper data: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def close_to_printed(actual: float, printed: str) -> bool:
    """Accept only differences explainable by four-decimal rounding."""

    return math.isclose(actual, float(printed), rel_tol=0.0, abs_tol=5.1e-5)


def check_derived_data() -> None:
    """Independently recompute every aggregate from the selected run-level CSV."""

    runs = csv_rows("run_level.csv")
    keys = {(row["dataset"], row["seed"], row["method"]) for row in runs}
    if len(runs) != 54 or len(keys) != 54:
        raise AssertionError(
            f"Expected 54 unique main runs, found {len(runs)} rows/{len(keys)} keys."
        )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        grouped[(row["dataset"], row["method"])].append(row)
    aggregates = csv_rows("aggregate.csv")
    if len(aggregates) != 18:
        raise AssertionError(f"Expected 18 aggregate rows, found {len(aggregates)}.")
    metric_map = {
        "accuracy_mean": ("accuracy", statistics.mean),
        "accuracy_std": ("accuracy", statistics.stdev),
        "worst_tpr_mean": ("worst_tpr", statistics.mean),
        "worst_tpr_std": ("worst_tpr", statistics.stdev),
        "mean_tpr_mean": ("mean_tpr", statistics.mean),
        "mean_tpr_std": ("mean_tpr", statistics.stdev),
    }
    for row in aggregates:
        key = (row["dataset"], row["method"])
        group = grouped[key]
        if len(group) != 3 or int(row["seeds"]) != 3:
            raise AssertionError(f"Aggregate {key} does not contain exactly three seeds.")
        for output_field, (input_field, reducer) in metric_map.items():
            recomputed = reducer([float(item[input_field]) for item in group])
            if not math.isclose(
                recomputed, float(row[output_field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise AssertionError(
                    f"Aggregate CSV mismatch for {key} {output_field}: "
                    f"stored={row[output_field]}, recomputed={recomputed}"
                )

    attack_fields = (
        "fedmia_loss",
        "fedmia_cosine",
        "fedmia_joint",
        "nasr_passive",
        "rmia",
        "quantile_mia",
    )
    attack_rows = csv_rows("attack_aggregate.csv")
    if len(attack_rows) != 108:
        raise AssertionError(f"Expected 108 attack aggregates, found {len(attack_rows)}.")
    for row in attack_rows:
        key = (row["dataset"], row["method"])
        attack = row["attack"]
        if attack not in attack_fields:
            raise AssertionError(f"Unexpected attack aggregate {attack}.")
        values = [float(item[attack]) for item in grouped[key]]
        for output_field, reducer in (
            ("tpr_mean", statistics.mean),
            ("tpr_std", statistics.stdev),
        ):
            recomputed = reducer(values)
            if not math.isclose(
                recomputed, float(row[output_field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise AssertionError(
                    f"Attack aggregate mismatch for {key + (attack,)} {output_field}."
                )

    accounting = csv_rows("privacy_accounting.csv")
    if len(accounting) != 36 or any(
        row["formal_dp_enabled"].lower() != "true" for row in accounting
    ):
        raise AssertionError("Expected 36 formally enabled privacy-accounting rows.")
    accounting_summary = csv_rows("privacy_accounting_summary.csv")
    if len(accounting_summary) != 4:
        raise AssertionError("Expected four privacy-accounting method/scope rows.")
    ablations = csv_rows("ablation.csv")
    variants = {row["variant"] for row in ablations}
    if len(ablations) != 7 or len(variants) != 7:
        raise AssertionError("Expected seven unique VEIL component ablations.")


def check_result_tables(text: str) -> None:
    """Cross-check every manuscript result cell against validated CSV evidence."""

    aggregates = {
        (row["dataset"], row["method"]): row
        for row in csv_rows("aggregate.csv")
    }
    dataset_names = {
        "Flowers102": "flowers",
        "Caltech101": "caltech101",
        "DTD": "dtd",
    }
    method_names = {
        "No defense": "FedAvg",
        "Prompt-DP": "Prompt-DP",
        "HAMP": "HAMP",
        r"\method{}": "VEIL",
    }
    fedavg_block = re.search(
        r"\\begin\{tabular\}\{llccc\}(.*?)\\end\{tabular\}",
        text,
        flags=re.DOTALL,
    )
    if fedavg_block is None:
        raise AssertionError("Could not locate the FedAvg result table.")
    pattern = re.compile(
        r"^(Flowers102|Caltech101|DTD) & (No defense|Prompt-DP|HAMP|\\method\{\}) "
        r"& ([0-9.]+) \$\\pm\$ ([0-9.]+) "
        r"& ([0-9.]+) \$\\pm\$ ([0-9.]+) "
        r"& ([0-9.]+) \$\\pm\$ ([0-9.]+)\\\\$",
        flags=re.MULTILINE,
    )
    parsed = pattern.findall(fedavg_block.group(1))
    if len(parsed) != 12:
        raise AssertionError(f"Expected 12 FedAvg table rows, parsed {len(parsed)}.")
    fields = (
        "accuracy_mean",
        "accuracy_std",
        "worst_tpr_mean",
        "worst_tpr_std",
        "mean_tpr_mean",
        "mean_tpr_std",
    )
    for dataset_label, method_label, *printed in parsed:
        key = (dataset_names[dataset_label], method_names[method_label])
        evidence = aggregates.get(key)
        if evidence is None:
            raise AssertionError(f"No aggregate evidence for {key}.")
        for field, value in zip(fields, printed):
            if not close_to_printed(float(evidence[field]), value):
                raise AssertionError(
                    f"FedAvg table mismatch for {key} {field}: "
                    f"paper={value}, evidence={evidence[field]}"
                )

    attacks = {
        (row["dataset"], row["method"], row["attack"]): float(row["tpr_mean"])
        for row in csv_rows("attack_aggregate.csv")
    }
    attack_order = (
        "fedmia_loss",
        "fedmia_cosine",
        "fedmia_joint",
        "nasr_passive",
        "rmia",
        "quantile_mia",
    )
    private_block = re.search(
        r"\\begin\{tabular\}\{llcccccc\}(.*?)\\end\{tabular\}",
        text,
        flags=re.DOTALL,
    )
    if private_block is None:
        raise AssertionError("Could not locate the private-mechanism result table.")
    private_pattern = re.compile(
        r"^(Flowers102|Caltech101|DTD) & (DP-FPL|FedASK) & "
        r"([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & "
        r"([0-9.]+) & ([0-9.]+)\\\\$",
        flags=re.MULTILINE,
    )
    private_rows = private_pattern.findall(private_block.group(1))
    if len(private_rows) != 6:
        raise AssertionError(
            f"Expected 6 private-mechanism table rows, parsed {len(private_rows)}."
        )
    for dataset_label, method, *printed in private_rows:
        dataset = dataset_names[dataset_label]
        for attack, value in zip(attack_order, printed):
            actual = attacks[(dataset, method, attack)]
            if not close_to_printed(actual, value):
                raise AssertionError(
                    f"Private table mismatch for {(dataset, method, attack)}: "
                    f"paper={value}, evidence={actual}"
                )

    accounting = {
        (row["method"], row["scope"]): row
        for row in csv_rows("privacy_accounting_summary.csv")
    }
    expected_accounting = {
        ("Prompt-DP", "prompt update"): ("Prompt-DP", "epsilon"),
        ("DP-FPL", "local"): ("DP-FPL", "local_epsilon"),
        ("DP-FPL", "global"): ("DP-FPL", "global_epsilon"),
        ("FedASK", "local"): ("FedASK", "epsilon"),
    }
    accounting_block = re.search(
        r"\\begin\{tabular\}\{llcc\}(.*?)\\end\{tabular\}",
        text,
        flags=re.DOTALL,
    )
    if accounting_block is None:
        raise AssertionError("Could not locate the privacy-accounting table.")
    accounting_pattern = re.compile(
        r"^(Prompt-DP|DP-FPL|FedASK) & (prompt update|local|global) & "
        r"([0-9.]+)(?:--([0-9.]+))? & \$10\^\{-5\}\$\\\\$",
        flags=re.MULTILINE,
    )
    accounting_rows = accounting_pattern.findall(accounting_block.group(1))
    if len(accounting_rows) != 4:
        raise AssertionError(
            f"Expected 4 privacy-accounting rows, parsed {len(accounting_rows)}."
        )
    for method, scope, printed_min, printed_max in accounting_rows:
        source_key = expected_accounting[(method, scope)]
        evidence = accounting[source_key]
        if not math.isclose(
            float(evidence["epsilon_min"]),
            float(printed_min),
            rel_tol=0.0,
            abs_tol=5.1e-3,
        ):
            raise AssertionError(f"Accounting minimum mismatch for {source_key}.")
        shown_max = float(printed_max or printed_min)
        if not math.isclose(
            float(evidence["epsilon_max"]), shown_max, rel_tol=0.0, abs_tol=5.1e-3
        ):
            raise AssertionError(f"Accounting maximum mismatch for {source_key}.")
        if not math.isclose(float(evidence["delta"]), 1e-5, rel_tol=0.0, abs_tol=0.0):
            raise AssertionError(f"Accounting delta mismatch for {source_key}.")


def main() -> None:
    source = PAPER.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    text = uncommented(source)
    check_braces(text)
    check_environments(text)
    check_citations(text, bib)
    packages = set(re.findall(r"\\usepackage(?:\[[^]]*\])?\{([^}]+)\}", text))
    bad_packages = packages & FORBIDDEN_PACKAGES
    if bad_packages:
        raise AssertionError(f"Forbidden AAAI packages: {sorted(bad_packages)}")
    for command in FORBIDDEN_COMMANDS:
        if re.search(re.escape(command) + r"\b", text):
            raise AssertionError(f"Forbidden AAAI command: {command}")
    if re.search(r"\\v(?:space|skip)\s*\{\s*-", text):
        raise AssertionError("Negative vertical spacing is forbidden by the author kit.")
    if "Anonymous Submission" not in text or "\\usepackage[submission]{aaai2027}" not in text:
        raise AssertionError("The manuscript is not configured as an anonymous submission.")
    if "& --" in text or "still being populated" in text:
        raise AssertionError("The manuscript still contains experimental placeholders.")
    check_figures(text)
    check_derived_data()
    check_result_tables(text)
    checklist_source = CHECKLIST.read_text(encoding="utf-8")
    check_braces(uncommented(checklist_source))
    check_environments(uncommented(checklist_source))
    checklist = checklist_source.split(
        "% The questions start here", maxsplit=1
    )[1]
    question_count = len(re.findall(r"\\question\{", checklist))
    answer_count = len(
        re.findall(r"^\s*(?:yes|no|partial|NA)\s*$", checklist, flags=re.MULTILINE)
    )
    if question_count != answer_count:
        raise AssertionError(
            f"Reproducibility checklist has {question_count} questions but "
            f"{answer_count} answers."
        )
    print(
        "Static AAAI syntax, citation, package, placeholder, figure, result-data, "
        "and reproducibility-checklist checks passed."
    )


if __name__ == "__main__":
    main()
