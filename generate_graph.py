import json
import dataclasses
import math

from scipy.stats import mannwhitneyu, pmean
import numpy as np
import matplotlib.pyplot as plt 

@dataclasses.dataclass
class Result:
    bm_name: str
    bm_values: list[float]

@dataclasses.dataclass
class Diff:
    bm_name: str
    bm_diff: float

def parse_json(filename: str) -> dict[str, Result]:
    results = {}
    with open(filename) as fp:
        read = json.load(fp)
        for benchmark in read["benchmarks"]:
            bm_name = benchmark["metadata"]["name"]
            bm_values = []
            for run in benchmark["runs"]:
                bm_values += run.get("values", [])
            results[bm_name] = Result(bm_name, bm_values)
    return results

def calculate_significance(results1: dict[str, Result], results2: dict[str, Result]):
    nan = float('nan')
    final_result = {}
    for bm_name, base in results1.items():
        base_values = base.bm_values
        other_values = results2[bm_name].bm_values
        _, p_value = (mannwhitneyu(base_values, other_values))
        bm_diff = None
        # Significant
        if p_value < 0.05:
            # Geometric mean
            bm_diff =  pmean(base_values, p=0) / (pmean(other_values, p=0))
        # Not significant
        else:
            bm_diff = nan
        final_result[bm_name] = Diff(bm_name, bm_diff)
    return final_result

        

def main():
    without_opt = parse_json("./bm-20240629-3.14.0a0-118726c-JIT/bm-20240629-linux-x86_64-Fidget%2dSpinner-optimizer_off-3.14.0a0-118726c.json")
    with_opt = parse_json("./bm-20240629-3.14.0a0-e6543da-JIT/bm-20240629-linux-x86_64-python-e6543daf12051e9c660a-3.14.0a0-e6543da.json")
    diffs = list(calculate_significance(without_opt, with_opt).items())
    not_significant = [diff for diff in diffs if math.isnan(diff[1].bm_diff)]
    significant = [diff for diff in diffs if not math.isnan(diff[1].bm_diff)]
    # Sort by biggest difference first
    significant.sort(key=lambda a: a[1].bm_diff)
    significant.reverse()
    x_labels = [diff[0] for diff in significant]
    y_labels = [diff[1].bm_diff for diff in significant]
    f, ax = plt.subplots(1)
    ax.bar(x_labels, y_labels, color ='blue')
    ax.set_ylim(bottom=0.95)
    ax.axhline(y=1.0, color='r', linestyle='-')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.40)
    plt.show()
    




if __name__ == "__main__":
    main()
