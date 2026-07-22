import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


DEFAULT_DIR_A = Path(
    r"C:\Users\wangy\Desktop\code\SNNTransformer\Spikingformer\cifar10"
    r"\C_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_tensor_distributon"
)
DEFAULT_DIR_B = Path(
    r"C:\Users\wangy\Desktop\code\SNNTransformer\Spikingformer\cifar10"
    r"\C_model_4layers_baseline_ILIF01_count_to_adaptive_back_four_online_noatten_tensor_distributon"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\wangy\Desktop\code\SNNTransformer\Spikingformer\cifar10"
    r"\C_model_count_vs_four_online_noatten_tensor_distribution_compare"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine same-name PNG files from two folders into side-by-side comparison images."
    )
    parser.add_argument("--dir-a", type=Path, default=DEFAULT_DIR_A, help="First PNG folder.")
    parser.add_argument("--dir-b", type=Path, default=DEFAULT_DIR_B, help="Second PNG folder.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--label-a", default="count", help="Title shown above images from dir-a.")
    parser.add_argument("--label-b", default="four_online_noatten", help="Title shown above images from dir-b.")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    return parser.parse_args()


def list_pngs(folder):
    if not folder.exists():
        raise FileNotFoundError("Folder does not exist: {}".format(folder))
    if not folder.is_dir():
        raise NotADirectoryError("Not a folder: {}".format(folder))
    return {path.name: path for path in folder.glob("*.png")}


def figure_size_for_images(image_a, image_b, dpi):
    max_height = max(image_a.shape[0], image_b.shape[0])
    total_width = image_a.shape[1] + image_b.shape[1]
    width_inches = min(max(total_width / dpi, 8.0), 18.0)
    height_inches = min(max(max_height / dpi + 0.8, 4.0), 10.0)
    return width_inches, height_inches


def save_comparison(name, path_a, path_b, output_path, label_a, label_b, dpi):
    image_a = mpimg.imread(path_a)
    image_b = mpimg.imread(path_b)
    fig_size = figure_size_for_images(image_a, image_b, dpi)

    fig, axes = plt.subplots(1, 2, figsize=fig_size, constrained_layout=True)
    for ax, image, label in ((axes[0], image_a, label_a), (axes[1], image_b, label_b)):
        ax.imshow(image)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(name, fontsize=12)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    pngs_a = list_pngs(args.dir_a)
    pngs_b = list_pngs(args.dir_b)

    common_names = sorted(set(pngs_a) & set(pngs_b))
    only_a = sorted(set(pngs_a) - set(pngs_b))
    only_b = sorted(set(pngs_b) - set(pngs_a))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in common_names:
        output_path = args.out_dir / name
        save_comparison(
            name=name,
            path_a=pngs_a[name],
            path_b=pngs_b[name],
            output_path=output_path,
            label_a=args.label_a,
            label_b=args.label_b,
            dpi=args.dpi,
        )

    print("dir_a: {}".format(args.dir_a))
    print("dir_b: {}".format(args.dir_b))
    print("out_dir: {}".format(args.out_dir))
    print("matched_pngs: {}".format(len(common_names)))
    print("only_in_dir_a: {}".format(len(only_a)))
    print("only_in_dir_b: {}".format(len(only_b)))
    if only_a:
        print("only_in_dir_a_names: {}".format(", ".join(only_a)))
    if only_b:
        print("only_in_dir_b_names: {}".format(", ".join(only_b)))


if __name__ == "__main__":
    main()
