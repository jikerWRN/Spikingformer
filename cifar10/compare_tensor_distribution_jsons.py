import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DEFAULT_DIR_A = Path(
    r"/home/wangyufei/code/SNNTransformer/Spikingformer/cifar10"
    r"/D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_kernel3_tensor_distributon"
)
DEFAULT_DIR_B = Path(
    r"/home/wangyufei/code/SNNTransformer/Spikingformer/cifar10"
    r"/D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_kernel3_tensor_distributon"
)
DEFAULT_OUTPUT_DIR = Path(
    r"/home/wangyufei/code/SNNTransformer/Spikingformer/cifar10"
    r"/C_kernel3_vs_add_internel_kernel3_tensor_distribution_json_compare"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot same-name tensor-distribution JSON files from two folders on one figure."
    )
    parser.add_argument("--dir-a", type=Path, default=DEFAULT_DIR_A, help="First JSON folder.")
    parser.add_argument("--dir-b", type=Path, default=DEFAULT_DIR_B, help="Second JSON folder.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--label-a", default="kernel3", help="Legend label for dir-a.")
    parser.add_argument("--label-b", default="add_internal_kernel3", help="Legend label for dir-b.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Plot raw histogram counts instead of normalized probability density.",
    )
    return parser.parse_args()


def list_jsons(folder):
    if not folder.exists():
        raise FileNotFoundError("Folder does not exist: {}".format(folder))
    if not folder.is_dir():
        raise NotADirectoryError("Not a folder: {}".format(folder))
    return {path.name: path for path in folder.glob("*.json")}


def load_distribution(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    histogram = data.get("histogram")
    if not histogram:
        return {
            "data": data,
            "counts": [],
            "edges": [],
            "y": [],
        }

    counts = [float(v) for v in histogram["counts"]]
    edges = [float(v) for v in histogram["bin_edges"]]
    if len(edges) != len(counts) + 1:
        raise ValueError("Invalid histogram in {}".format(path))

    return {
        "data": data,
        "counts": counts,
        "edges": edges,
        "y": counts,
    }


def histogram_y(counts, edges, raw_counts):
    if raw_counts:
        return counts

    total = sum(counts)
    if total <= 0:
        return [0.0 for _ in counts]

    density = []
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        width = right - left
        density.append(count / total / width if width > 0 else 0.0)
    return density


def quantile_value(dist, key):
    quantiles = dist["data"].get("quantiles", {})
    return quantiles.get(key)


def stats_text(label, dist):
    data = dist["data"]
    return (
        "{}\n"
        "n={}\n"
        "mean={:.6g}\n"
        "std={:.6g}\n"
        "q50={:.6g}\n"
        "q99={:.6g}"
    ).format(
        label,
        data.get("numel_sampled", 0),
        data.get("mean", 0.0),
        data.get("std", 0.0),
        quantile_value(dist, "0.50") or 0.0,
        quantile_value(dist, "0.99") or 0.0,
    )


def plot_distribution_pair(name, path_a, path_b, output_path, label_a, label_b, dpi, raw_counts):
    dist_a = load_distribution(path_a)
    dist_b = load_distribution(path_b)
    y_a = histogram_y(dist_a["counts"], dist_a["edges"], raw_counts)
    y_b = histogram_y(dist_b["counts"], dist_b["edges"], raw_counts)

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    if y_a:
        ax.stairs(y_a, dist_a["edges"], color="#2563eb", linewidth=1.8, label=label_a)
    if y_b:
        ax.stairs(y_b, dist_b["edges"], color="#dc2626", linewidth=1.8, label=label_b)

    median_a = quantile_value(dist_a, "0.50")
    median_b = quantile_value(dist_b, "0.50")
    if median_a is not None:
        ax.axvline(float(median_a), color="#2563eb", linestyle="--", linewidth=1.0, alpha=0.9)
    if median_b is not None:
        ax.axvline(float(median_b), color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.9)

    ylabel = "count" if raw_counts else "probability density"
    ax.set_xlabel("state value")
    ax.set_ylabel(ylabel)
    ax.set_title(name)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.text(
        0.99,
        0.98,
        stats_text(label_a, dist_a) + "\n\n" + stats_text(label_b, dist_b),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    jsons_a = list_jsons(args.dir_a)
    jsons_b = list_jsons(args.dir_b)

    common_names = sorted(set(jsons_a) & set(jsons_b))
    only_a = sorted(set(jsons_a) - set(jsons_b))
    only_b = sorted(set(jsons_b) - set(jsons_a))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in common_names:
        output_path = args.out_dir / Path(name).with_suffix(".png").name
        plot_distribution_pair(
            name=name,
            path_a=jsons_a[name],
            path_b=jsons_b[name],
            output_path=output_path,
            label_a=args.label_a,
            label_b=args.label_b,
            dpi=args.dpi,
            raw_counts=args.raw_counts,
        )

    print("dir_a: {}".format(args.dir_a))
    print("dir_b: {}".format(args.dir_b))
    print("out_dir: {}".format(args.out_dir))
    print("matched_jsons: {}".format(len(common_names)))
    print("only_in_dir_a: {}".format(len(only_a)))
    print("only_in_dir_b: {}".format(len(only_b)))
    if only_a:
        print("only_in_dir_a_names: {}".format(", ".join(only_a)))
    if only_b:
        print("only_in_dir_b_names: {}".format(", ".join(only_b)))


if __name__ == "__main__":
    main()
