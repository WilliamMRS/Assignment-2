"""Detect connected hot-colored regions in the supplied heatmap images.

Run with the existing image_processing/venv Python. All coordinates and areas
refer to the original image resolution. This is color-threshold blob detection,
not an estimate of fixation counts or a scale-space (LoG) detector.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy import ndimage as ndi
from skimage import io, measure, util

HERE = Path(__file__).resolve().parent
CHOSEN_DIR = HERE / "chosen_images"
DEFAULT_IMAGES = ["hm1_1.png", "hm20_27.png", "hm20_9.png", "hm50_28.png"]


@dataclass(frozen=True)
class Parameters:
    threshold: float = 0.35
    smooth_sigma: float = 1.5
    min_area: int = 80

    def __post_init__(self):
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1 (exclusive)")
        if not np.isfinite(self.smooth_sigma) or self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be finite and nonnegative")
        if not isinstance(self.min_area, int) or self.min_area < 1:
            raise ValueError("min_area must be a positive integer")


def heat_score(image):
    """Red dominance in [0, 1]: clip(R - (G+B)/2, 0, 1).

    Red scores 1, yellow 0.5, and white/gray/black/green/blue score 0.
    This is a color-selection rule, not recovery of the source heat values.
    Composite RGBA inputs over white, matching the supplied slide backgrounds.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA heatmap image")
    rgb = util.img_as_float(image)
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Image values must be finite and in [0, 1]")
    if rgb.shape[2] == 4:
        rgb = rgb[..., :3] * rgb[..., 3:] + (1 - rgb[..., 3:])
    return np.clip(rgb[..., 0] - (rgb[..., 1] + rgb[..., 2]) / 2, 0, 1)


def detect_blobs(image, params=Parameters()):
    """Score -> Gaussian smooth -> threshold -> 8-connected components.

    Remove components below min_area and assign consecutive IDs in raster order.
    Return score, smoothed score, retained labels, and one statistics row per blob.
    Neighboring pixels connect across edges or corners. Border blobs are retained
    and flagged because their visible area may be truncated.
    """
    score = heat_score(image)
    smoothed = ndi.gaussian_filter(score, params.smooth_sigma, mode="reflect")
    labels, _ = ndi.label(smoothed >= params.threshold, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())
    keep = areas >= params.min_area
    keep[0] = False
    mapping = np.zeros(len(areas), dtype=np.int32)
    mapping[keep] = np.arange(1, np.count_nonzero(keep) + 1)
    labels = mapping[labels]
    rows = []
    height, width = labels.shape
    for region in measure.regionprops(labels):
        y, x = region.centroid
        y0, x0, y1, x1 = region.bbox
        rows.append(dict(
            blob_id=int(region.label), area_px=int(region.area),
            center_x_px=float(x), center_y_px=float(y),
            equivalent_radius_px=float(np.sqrt(region.area / np.pi)),
            bbox_x0_px=int(x0), bbox_y0_px=int(y0),
            bbox_x1_px=int(x1), bbox_y1_px=int(y1),
            touches_border=bool(x0 == 0 or y0 == 0 or x1 == width or y1 == height),
        ))
    return dict(score=score, smoothed=smoothed, labels=labels, blobs=rows)


def summarize(name, image, result):
    areas = np.array([b["area_px"] for b in result["blobs"]])
    return dict(image=name, width_px=image.shape[1], height_px=image.shape[0],
                blob_count=len(areas), total_area_px=int(areas.sum()),
                mean_area_px=float(areas.mean()) if len(areas) else 0.0,
                min_area_px=int(areas.min()) if len(areas) else 0,
                max_area_px=int(areas.max()) if len(areas) else 0,
                coverage_percent=100 * float(areas.sum()) / image.shape[0] / image.shape[1],
                border_blob_count=sum(b["touches_border"] for b in result["blobs"]))


def overlay(ax, image, result, title):
    ax.imshow(image)
    for blob in result["blobs"]:
        x, y = blob["center_x_px"], blob["center_y_px"]
        ax.add_patch(Circle((x, y), blob["equivalent_radius_px"],
                            fill=False, edgecolor="#101010", linewidth=2.5))
        ax.add_patch(Circle((x, y), blob["equivalent_radius_px"],
                            fill=False, edgecolor="#00ffff", linewidth=1.2))
        ax.text(x, y, str(blob["blob_id"]), color="black", fontsize=8,
                ha="center", va="center", bbox=dict(facecolor="cyan", alpha=.85,
                edgecolor="none", pad=1))
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def save_fig(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(name, image, result, params, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"1. Original heatmap ({name})")
    im = axes[0, 1].imshow(result["smoothed"], cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title(f"2. Red-dominance score, Gaussian sigma = {params.smooth_sigma} px")
    fig.colorbar(im, ax=axes[0, 1], shrink=.75, label="Smoothed color score")
    axes[1, 0].imshow(result["labels"] > 0, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"3. Score >= {params.threshold}; area >= {params.min_area} px")
    overlay(axes[1, 1], image, result,
            f"4. {len(result['blobs'])} connected blobs, equivalent-area circles + IDs")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.tight_layout()
    save_fig(fig, out, "1_pipeline.png")


def plot_detections(images, results, summaries, out):
    fig, axes = plt.subplots(len(images), 2, figsize=(15, 4.4 * len(images)), squeeze=False)
    for (name, image), result, summary, row in zip(images, results, summaries, axes):
        row[0].imshow(image)
        row[0].set_title(f"{name}: original", fontsize=11)
        row[0].set_axis_off()
        overlay(row[1], image, result,
                f"{summary['blob_count']} blobs | mean area {summary['mean_area_px']:.0f} px | "
                f"coverage {summary['coverage_percent']:.2f}%")
    fig.suptitle("Hot-region detections at original resolution\n"
                 "Circle area equals segmented area; IDs match blob_statistics.csv", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_fig(fig, out, "2_detections.png")


def plot_statistics(images, results, out):
    fig, axes = plt.subplots(len(images), 1, figsize=(12, 2.6 * len(images)), squeeze=False)
    for (name, _), result, (ax,) in zip(images, results, axes):
        ax.set_axis_off()
        ax.set_title(name + " (x rightwards, y downwards; origin at top left)", fontsize=11)
        rows = [[b["blob_id"], b["area_px"], f"{b['center_x_px']:.1f}",
                 f"{b['center_y_px']:.1f}", f"{b['equivalent_radius_px']:.1f}",
                 "yes" if b["touches_border"] else "no"] for b in result["blobs"]]
        # For a large custom input, CSV contains all rows; keep this figure readable.
        shown = rows[:10] or [["-", 0, "-", "-", "-", "-"]]
        table = ax.table(cellText=shown, bbox=[0, 0, 1, .85],
                         colLabels=["ID", "Area (px)", "Center x (px)", "Center y (px)",
                                    "Eq. radius (px)", "At border"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        if len(rows) > 10:
            ax.text(0, -.1, f"Showing first 10 of {len(rows)} blobs; see CSV for all rows.")
    fig.tight_layout()
    save_fig(fig, out, "3_statistics.png")


def parameter_sweep(images, baseline, out):
    """Change only one parameter at a time; record results for every image."""
    sweeps = [("threshold", [0.20, 0.35, 0.60]),
              ("smooth_sigma", [0.0, 1.5, 4.0]),
              ("min_area", [80, 300, 1500])]
    sweep_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (parameter, values) in enumerate(sweeps):
        for name, image in images:
            summaries = []
            for value in values:
                params = replace(baseline, **{parameter: value})
                result = detect_blobs(image, params)
                summary = summarize(name, image, result)
                summaries.append(summary)
                sweep_rows.append(dict(varied_parameter=parameter, **asdict(params), **summary))
            axes[0, col].plot(values, [s["blob_count"] for s in summaries], "o-", label=name)
            axes[1, col].plot(values, [s["coverage_percent"] for s in summaries], "o-")
        for row in range(2):
            axes[row, col].set_xlabel(parameter)
            axes[row, col].set_xticks(values)
            axes[row, col].grid(alpha=.25)
        axes[0, col].set_ylabel("Blob count")
        axes[1, col].set_ylabel("Image area covered (%)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("One parameter changed at a time\n"
                 f"Baseline: threshold={baseline.threshold}, sigma={baseline.smooth_sigma}, "
                 f"minimum area={baseline.min_area}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .91))
    save_fig(fig, out, "4_parameter_sweep.png")

    # Show merging/splitting on the mixed image, small-blob loss on the sparse image.
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for index, (row, (parameter, values)) in enumerate(zip(axes, sweeps)):
        example_index = 2 if index == 0 and len(images) >= 3 else len(images) - 1
        name, image = images[example_index]
        for ax, value in zip(row, values):
            params = replace(baseline, **{parameter: value})
            result = detect_blobs(image, params)
            overlay(ax, image, result,
                    f"{name}: {parameter} = {value}\n{len(result['blobs'])} blobs")
    fig.suptitle("Parameter effects (other parameters held at baseline)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=2)
    save_fig(fig, out, "5_parameter_examples.png")
    return sweep_rows


BLOB_FIELDS = ["image", "blob_id", "area_px", "center_x_px", "center_y_px",
               "equivalent_radius_px", "bbox_x0_px", "bbox_y0_px", "bbox_x1_px",
               "bbox_y1_px", "touches_border"]


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path,
                        help="Optional RGB/RGBA heatmaps instead of the four selected examples")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    paths = args.images or [CHOSEN_DIR / name for name in DEFAULT_IMAGES]
    images = [(p.stem, io.imread(p)) for p in paths]
    if len({name for name, _ in images}) != len(images):
        parser.error("Input image stems must be unique for unambiguous statistics")
    args.output.mkdir(parents=True, exist_ok=True)
    params = Parameters()
    results = [detect_blobs(image, params) for _, image in images]
    summaries = [summarize(name, image, result)
                 for (name, image), result in zip(images, results)]
    blobs = [dict(image=name, **blob) for (name, _), result in zip(images, results)
             for blob in result["blobs"]]
    write_csv(args.output / "image_summary.csv", summaries)
    write_csv(args.output / "blob_statistics.csv", blobs, BLOB_FIELDS)
    (args.output / "parameters.json").write_text(json.dumps(dict(
        parameters=asdict(params), connectivity=8, gaussian_boundary_mode="reflect",
        score="clip(R - (G+B)/2, 0, 1)",
        positions="Unweighted centroid; x=column, y=row; zero-based pixels",
        bounding_boxes="x0/y0 inclusive; x1/y1 exclusive",
        images=[str(p.resolve()) for p in paths],
    ), indent=2) + "\n")
    plot_pipeline(images[0][0], images[0][1], results[0], params, args.output)
    plot_detections(images, results, summaries, args.output)
    plot_statistics(images, results, args.output)
    sweep_rows = parameter_sweep(images, params, args.output)
    write_csv(args.output / "parameter_sweep.csv", sweep_rows)
    for summary in summaries:
        print(f"{summary['image']:12s}: {summary['blob_count']:2d} blobs; "
              f"mean area {summary['mean_area_px']:.1f} px; "
              f"coverage {summary['coverage_percent']:.2f}%")
        for blob in [b for b in blobs if b["image"] == summary["image"]]:
            print(f"  ID {blob['blob_id']:2d}: area={blob['area_px']:5d} px, "
                  f"center=({blob['center_x_px']:.1f}, {blob['center_y_px']:.1f}), "
                  f"equivalent radius={blob['equivalent_radius_px']:.1f} px")
    print(f"\nFigures and CSV statistics: {args.output.resolve()}")


if __name__ == "__main__":
    main()
