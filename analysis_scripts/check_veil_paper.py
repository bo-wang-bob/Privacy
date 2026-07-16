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
        "DP-FPL": "DP-FPL",
        "FedASK": "FedASK",
    }
    fields = (
        "accuracy_mean",
        "accuracy_std",
        "worst_tpr_mean",
        "worst_tpr_std",
        "mean_tpr_mean",
        "mean_tpr_std",
    )

    def table_block(label: str) -> str:
        label_token = rf"\label{{{label}}}"
        label_index = text.find(label_token)
        if label_index < 0:
            raise AssertionError(f"Could not locate table label {label}.")
        plain_start = text.rfind(r"\begin{table}", 0, label_index)
        star_start = text.rfind(r"\begin{table*}", 0, label_index)
        start = max(plain_start, star_start)
        if start < 0:
            raise AssertionError(f"Could not locate table start for {label}.")
        end_token = r"\end{table*}" if start == star_start else r"\end{table}"
        end = text.find(end_token, label_index)
        if end < 0:
            raise AssertionError(f"Could not locate table end for {label}.")
        return text[start : end + len(end_token)]

    def parse_grouped_table(
        block: str,
        allowed_methods: set[str],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        current_dataset: str | None = None
        shaded = False
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line == r"\rowcolor{veilshade}":
                shaded = True
                continue
            dataset_match = re.search(
                r"\\multirow\{\d+\}\{\*\}\{(Flowers102|Caltech101|DTD)\}",
                line,
            )
            if dataset_match:
                current_dataset = dataset_match.group(1)
            if "&" not in line or not line.endswith(r"\\"):
                continue
            cells = [cell.strip() for cell in line[:-2].split("&")]
            if len(cells) != 5:
                shaded = False
                continue
            method_label = cells[1]
            if method_label not in allowed_methods:
                shaded = False
                continue
            if current_dataset is None:
                raise AssertionError(f"Missing multirow dataset before {line}")
            printed: list[str] = []
            for metric_cell in cells[2:]:
                numbers = re.findall(r"[0-9]+\.[0-9]+", metric_cell)
                if len(numbers) != 2:
                    raise AssertionError(f"Expected mean and std in cell {metric_cell}")
                printed.extend(numbers)
            rows.append(
                {
                    "dataset_label": current_dataset,
                    "method_label": method_label,
                    "printed": printed,
                    "bold": tuple(r"\textbf{" in cell for cell in cells[2:]),
                    "shaded": shaded,
                }
            )
            shaded = False
        return rows

    def validate_grouped_table(
        label: str,
        allowed_methods: set[str],
        expected_count: int,
    ) -> None:
        block = table_block(label)
        if len(re.findall(r"\\multirow\{\d+\}\{\*\}\{", block)) != 3:
            raise AssertionError(f"{label} must merge its three dataset columns.")
        parsed = parse_grouped_table(block, allowed_methods)
        if len(parsed) != expected_count:
            raise AssertionError(
                f"Expected {expected_count} rows in {label}, parsed {len(parsed)}."
            )
        table_methods = {method_names[item] for item in allowed_methods}
        metric_fields = ("accuracy_mean", "worst_tpr_mean", "mean_tpr_mean")
        reducers = (max, min, min)
        for row in parsed:
            dataset = dataset_names[str(row["dataset_label"])]
            method = method_names[str(row["method_label"])]
            evidence = aggregates[(dataset, method)]
            for field, value in zip(fields, row["printed"]):
                if not close_to_printed(float(evidence[field]), str(value)):
                    raise AssertionError(
                        f"{label} mismatch for {(dataset, method)} {field}: "
                        f"paper={value}, evidence={evidence[field]}"
                    )
            expected_bold = []
            for field, reducer in zip(metric_fields, reducers):
                values = [
                    float(aggregates[(dataset, candidate)][field])
                    for candidate in table_methods
                ]
                best = reducer(values)
                expected_bold.append(
                    math.isclose(
                        float(evidence[field]), best, rel_tol=0.0, abs_tol=5.1e-5
                    )
                )
            if tuple(expected_bold) != row["bold"]:
                raise AssertionError(
                    f"Incorrect bolding in {label} for {(dataset, method)}: "
                    f"shown={row['bold']}, expected={tuple(expected_bold)}"
                )
            if bool(row["shaded"]) != (method == "VEIL"):
                raise AssertionError(
                    f"Only VEIL rows must be shaded in {label}: {(dataset, method)}"
                )

    validate_grouped_table(
        "tab:fedavg",
        {"No defense", "Prompt-DP", "HAMP", r"\method{}"},
        12,
    )
    validate_grouped_table(
        "tab:private",
        {"DP-FPL", "FedASK", r"\method{}"},
        9,
    )

    ablation_source = {row["variant"]: row for row in csv_rows("ablation.csv")}
    ablation_names = {
        r"Full \method{}": "Full VEIL",
        "Individual anchor": "Individual anchor",
        "No echoes": "No echoes",
        "No prototype branch": "No prototype",
        "No upload smoothing": "No upload smoothing",
        "No output calib.": "No output tempering",
    }
    ablation_block = table_block("tab:ablation")
    parsed_ablation: list[tuple[str, list[str], tuple[bool, ...], bool]] = []
    shaded = False
    for raw_line in ablation_block.splitlines():
        line = raw_line.strip()
        if line == r"\rowcolor{veilshade}":
            shaded = True
            continue
        if "&" not in line or not line.endswith(r"\\"):
            continue
        cells = [cell.strip() for cell in line[:-2].split("&")]
        if len(cells) != 4 or cells[0] not in ablation_names:
            shaded = False
            continue
        values = []
        for cell in cells[1:]:
            numbers = re.findall(r"[0-9]+\.[0-9]+", cell)
            if len(numbers) != 1:
                raise AssertionError(f"Malformed ablation cell: {cell}")
            values.append(numbers[0])
        parsed_ablation.append(
            (
                ablation_names[cells[0]],
                values,
                tuple(r"\textbf{" in cell for cell in cells[1:]),
                shaded,
            )
        )
        shaded = False
    if len(parsed_ablation) != 6:
        raise AssertionError(f"Expected 6 focused ablations, found {len(parsed_ablation)}.")
    displayed = [ablation_source[name] for name, *_rest in parsed_ablation]
    ablation_fields = ("accuracy", "worst_tpr", "mean_tpr")
    ablation_reducers = (max, min, min)
    for name, printed, bold, row_shaded in parsed_ablation:
        evidence = ablation_source[name]
        for field, value in zip(ablation_fields, printed):
            if not close_to_printed(float(evidence[field]), value):
                raise AssertionError(f"Ablation mismatch for {name} {field}.")
        expected_bold = tuple(
            math.isclose(
                float(evidence[field]),
                reducer([float(item[field]) for item in displayed]),
                rel_tol=0.0,
                abs_tol=5.1e-5,
            )
            for field, reducer in zip(ablation_fields, ablation_reducers)
        )
        if bold != expected_bold:
            raise AssertionError(
                f"Incorrect ablation bolding for {name}: {bold} vs {expected_bold}."
            )
        if row_shaded != (name == "Full VEIL"):
            raise AssertionError(f"Only Full VEIL may be shaded in the ablation.")

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


def check_focused_structure(text: str) -> None:
    """Enforce the requested three-stage method and non-redundant result design."""

    method_match = re.search(
        r"\\section\{VEIL\}(.*?)\\section\{Experimental Methodology\}",
        text,
        flags=re.DOTALL,
    )
    if method_match is None:
        raise AssertionError("Could not isolate the VEIL method section.")
    method = method_match.group(1)
    subsections = re.findall(r"\\subsection\{([^}]+)\}", method)
    expected = [
        "Stage I: Private Semantic Geometry",
        "Stage II: Instance-Obscured Surrogate Learning",
        "Stage III: Structured Prompt Release",
    ]
    if subsections != expected:
        raise AssertionError(f"VEIL must have exactly three stages: {subsections}")
    if r"\begin{align}" in method or r"\begin{aligned}" in method:
        raise AssertionError("Method equations must not place multiple formulas per row.")
    algorithm = re.search(
        r"\\begin\{algorithmic\}\[1\](.*?)\\end\{algorithmic\}",
        method,
        flags=re.DOTALL,
    )
    if algorithm is None or len(re.findall(r"\\STATE\b", algorithm.group(1))) != 3:
        raise AssertionError("The VEIL algorithm must contain exactly three steps.")
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    if figures != ["figures/private_methods_attack_profile.pdf"]:
        raise AssertionError(f"Expected one focused attack figure, found {figures}.")
    if text.count(r"\rowcolor{veilshade}") < 7:
        raise AssertionError("VEIL rows are not consistently shaded in result tables.")


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
    check_focused_structure(text)
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
        "Static AAAI syntax, three-stage structure, table styling, citation, "
        "figure, result-data, and reproducibility-checklist checks passed."
    )


if __name__ == "__main__":
    main()
