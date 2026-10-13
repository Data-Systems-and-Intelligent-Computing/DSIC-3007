"""A short human-readable research result from the saved local analysis."""

from pathlib import Path


def _number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def _interval(values):
    return "unavailable" if not values else f"[{_number(values[0])}, {_number(values[1])}]"


def write_results(output, report):
    primary=report["primary_equal_domain"]
    lines=["# Research results", "", f"Stage: {report['stage']}. Role: {report['inference_role']}.",
           "", f"Complete events: {report['events']}; documents: {report['documents']}.",
           "", "## Paired primary endpoint", "",
           f"Equal scientific/code M2-minus-M1 estimate: {_number(primary.get('estimate'))} nats.",
           f"95% source-group bootstrap-t interval: {_interval(primary.get('ci'))}.",
           f"Valid draws: {primary['valid_replicates']}; degenerate draws: {primary['degenerate_replicates']}.",
           "", "## Scientific and code comparisons", "",
           "| Domain | CD M1 | CD M2 | Paired delta | 95% interval | Holm p |",
           "|---|---:|---:|---:|---|---:|"]
    adjusted=report["secondary_holm"].get("adjusted_p_values") or {}
    for domain in ("scientific","code"):
        paired=report["domain_paired_delta"].get(domain,{})
        absolute=report["absolute_discrimination"].get(domain,{})
        lines.append(f"| {domain} | {_number(absolute.get('M1'))} | {_number(absolute.get('M2'))} | "
                     f"{_number(paired.get('mean'))} | {_interval(paired.get('ci'))} | {_number(adjusted.get(domain))} |")
    lines.extend(["", "## Interpretation", "",
        "Inference is conditional on the frozen calibration and tested model, sources, context and intervention.",
        "Eight expert effects and repeated token positions are dependent measurements within their source units.",
        "An interval crossing zero expresses uncertainty; it does not establish equivalence.",
        ("Path B defers C4 outcomes, layers 0/7 and matched 2048; these results were not collected."
         if report.get("deferred_outcomes") else
         "C4, depth comparisons and context sensitivity are descriptive or exploratory."),
        "Pilot precision and main runtime are projections; actual main precision is reported from main data.",
        "", "Individual effects, controls, source identities and bootstrap draws remain in the raw and derived bundles."])
    (Path(output)/"research-results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
