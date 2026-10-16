"""
LBP (Local Binary Patterns), basic 8-neighbour method, implemented from scratch.

Steps covered (see assignment LBP tasks 1-7):
  1. lbp()                    -> LBP code for every pixel of a grayscale image
     plot_worked_example()    -> how one brick-wall neighbourhood becomes one code
  2. plot_lbp_images()        -> brick wall: original | LBP image
  3. lbp_histogram()          -> normalised 256-bin histogram of the LBP image
  4. plot_lbp_histograms()    -> brick-wall histogram (log scale, wide figure)
  5. plot_lbp_images()        -> brick wall plus forest, sea and face
  6. plot_lbp_histograms()    -> four histograms for comparison
  7. compare_histograms()     -> pattern types + distances to support the discussion

Optional extension: scene_classes() compares many Intel scene images.

Figures are written to image_processing/lbp/output/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io

# Paths are relative to this file, so the script runs from any working directory
HERE = Path(__file__).resolve().parent
INTEL_DIR = HERE.parent.parent / "data" / "intel-image-classification (1)" / "seg_test" / "seg_test"
CHOSEN_DIR = HERE / "chosen_images"
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# forest/sea: Intel Image Classification (150x150), brick: scikit-image data.brick(),
# face: facial-emotion-recognition set (subject 1, Neutral), cropped to the face.
IMAGES = [
    ("Texture: brick wall", "brick.png"),
    ("Natural scene: forest", "forest_20062.jpg"),
    ("Natural scene: sea", "sea_20106.jpg"),
    ("Face", "face_neutral.jpg"),
]

# Neighbour offsets (dy, dx), clockwise from the top-left. Neighbour i sets bit i (weight 2**i).
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


# ----------------------------------------------------------------------------
# Task 1 + 3: LBP image and its histogram
# ----------------------------------------------------------------------------
def load_gray(path):
    """Load an image as 8-bit grayscale (0-255), the classic input for LBP."""
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
        return np.round(img * 255).astype(np.uint8)
    return img.astype(np.uint8)


def lbp(img):
    """Basic 8-neighbour LBP.

    For each pixel, every neighbour that is >= the centre gives a 1, otherwise 0.
    The 8 bits are read clockwise from the top-left and summed with weights 2**i,
    giving a code in 0..255. The 1-pixel border has no full neighbourhood and is
    dropped, so the output is (H-2) x (W-2).
    """
    img = img.astype(np.int16)
    h, w = img.shape
    centre = img[1:-1, 1:-1]
    codes = np.zeros((h - 2, w - 2), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(OFFSETS):
        neighbour = img[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        codes |= (neighbour >= centre).astype(np.uint8) << bit
    return codes


def lbp_histogram(codes):
    """Normalised 256-bin histogram of an LBP image (sums to 1, so image size doesn't matter)."""
    hist = np.bincount(codes.ravel(), minlength=256).astype(float)
    return hist / hist.sum()


# ----------------------------------------------------------------------------
# Helpers for interpreting the 256 codes
# ----------------------------------------------------------------------------
def transitions(code):
    """Number of 0->1 / 1->0 changes when walking once around the circle of 8 bits."""
    bits = [(code >> i) & 1 for i in range(8)]
    return sum(bits[i] != bits[(i + 1) % 8] for i in range(8))


# "Uniform" codes have <= 2 transitions: one connected arc of 1s (58 of the 256 codes).
# Grouping them by number of 1s gives 9 rotation-invariant pattern types; all other codes -> 9.
PATTERN_TYPE = np.array([bin(c).count("1") if transitions(c) <= 2 else 9 for c in range(256)])
PATTERN_NAMES = [
    "0\nbright\nspot", "1\nline\nend", "2\ncorner", "3\ncorner", "4\nedge",
    "5\ncorner", "6\ncorner", "7\nline\nend", "8\nflat /\ndark spot", "9\nnon-\nuniform",
]


def pattern_type_histogram(hist):
    """Collapse a 256-bin LBP histogram into the 10 pattern types above."""
    return np.bincount(PATTERN_TYPE, weights=hist, minlength=10)


def chi_square(a, b, eps=1e-10):
    """Chi-square distance between two histograms (0 = identical). Standard for LBP."""
    return 0.5 * np.sum((a - b) ** 2 / (a + b + eps))


def entropy(hist):
    """Shannon entropy in bits: 0 = one code everywhere, 8 = all 256 codes equally common."""
    p = hist[hist > 0]
    return -np.sum(p * np.log2(p))


# ----------------------------------------------------------------------------
# Task 1: worked example of one code
# ----------------------------------------------------------------------------
def plot_worked_example(img, label):
    """Show how one 3x3 neighbourhood is thresholded and turned into a code."""
    codes = lbp(img)
    # Pick a deterministic, readable example: the middle pixel among those whose code is an edge
    ys, xs = np.nonzero(PATTERN_TYPE[codes] == 4)
    y, x = ys[len(ys) // 2] + 1, xs[len(xs) // 2] + 1  # +1: back to image coordinates
    patch = img[y - 1:y + 2, x - 1:x + 2].astype(int)
    c = patch[1, 1]

    bits = np.full((3, 3), -1)
    weights = np.full((3, 3), -1)
    for bit, (dy, dx) in enumerate(OFFSETS):
        bits[1 + dy, 1 + dx] = int(patch[1 + dy, 1 + dx] >= c)
        weights[1 + dy, 1 + dx] = 2 ** bit
    code = int(np.sum(bits[bits >= 0] * weights[weights >= 0]))
    assert code == codes[y - 1, x - 1]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(img, cmap="gray")
    axes[0].add_patch(plt.Rectangle((x - 6, y - 6), 12, 12, ec="red", fc="none", lw=2))
    axes[0].set_title(f"{label}\n(pixel at red box)")
    axes[0].axis("off")

    grids = [
        (patch, f"1. Grayscale values\n(centre = {c})", lambda v: str(v)),
        (bits, f"2. Neighbour >= {c} ?", lambda v: "" if v < 0 else str(v)),
        (weights, "3. Weights (clockwise)", lambda v: "" if v < 0 else str(v)),
    ]
    for ax, (grid, title, fmt) in zip(axes[1:], grids):
        ax.imshow(np.zeros((3, 3)), cmap="Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=16,
                    color="red" if (i, j) == (1, 1) else "black")
        ax.set_xticks(np.arange(-.5, 3), minor=True)
        ax.set_yticks(np.arange(-.5, 3), minor=True)
        ax.grid(which="minor", color="k", lw=1.5)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(title)

    bitstring = "".join(str(bits[1 + dy, 1 + dx]) for dy, dx in reversed(OFFSETS))
    fig.suptitle(f"LBP code = sum(bit x weight) = {bitstring}b = {code}", y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT_DIR / "1_worked_example.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Task 2 + 5: original and LBP images in a wide figure
# ----------------------------------------------------------------------------
def plot_lbp_images(images, filename="2_lbp_images.png"):
    results = []
    columns = min(2, len(images))
    rows = (len(images) + columns - 1) // columns
    fig = plt.figure(figsize=(14, 7), layout="constrained")
    groups = fig.subfigures(rows, columns, squeeze=False)
    for group, (label, img) in zip(groups.flat, images):
        codes = lbp(img)
        hist = lbp_histogram(codes)
        results.append((label, img, codes, hist))

        group.suptitle(f"{label} ({img.shape[1]}x{img.shape[0]})")
        row = group.subplots(1, 2)
        row[0].imshow(img, cmap="gray")
        row[0].set_title("Original")
        row[1].imshow(codes, cmap="gray", vmin=0, vmax=255)
        row[1].set_title("LBP image (code 0-255)")
        for ax in row[:2]:
            ax.axis("off")

    fig.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)
    return results


# ----------------------------------------------------------------------------
# Task 4 + 6: one demonstration histogram, then all four for comparison
# ----------------------------------------------------------------------------
def plot_lbp_histograms(results, filename="3_lbp_histograms.png"):
    columns = min(2, len(results))
    rows = (len(results) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(14, 4 if len(results) == 1 else 7), squeeze=False,
                             sharex=True, sharey=True)
    for ax, (label, img, codes, hist) in zip(axes.flat, results):
        ax.bar(np.arange(256), hist, width=1.0)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(-1, 256)
        ax.set_xticks([0, 31, 63, 127, 191, 255])
        ax.set_xlabel("LBP code")
        ax.set_ylabel("share of pixels (log)")
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_title(f"{label}\nLBP histogram - entropy {entropy(hist):.2f} bits")

        top = np.argsort(hist)[::-1][:3]
        print(f"{label:24s} uniform {pattern_type_histogram(hist)[:9].sum():5.1%}  "
              f"code255 {hist[255]:5.1%}  code0 {hist[0]:5.1%}  entropy {entropy(hist):.2f}  "
              f"top: " + ", ".join(f"{c} ({c:08b}) {hist[c]:.1%}" for c in top))
    for ax in list(axes.flat)[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Task 6 + 7: compare histograms between images
# ----------------------------------------------------------------------------
def compare_histograms(results):
    labels = [r[0] for r in results]
    hists = [r[3] for r in results]
    types = np.array([pattern_type_histogram(h) for h in hists])

    n = len(hists)
    dist = np.array([[chi_square(hists[i], hists[j]) for j in range(n)] for i in range(n)])

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), layout="constrained",
                             gridspec_kw=dict(width_ratios=[2, 1]))
    width = 0.8 / n
    for i, (label, t) in enumerate(zip(labels, types)):
        axes[0].bar(np.arange(10) + (i - (n - 1) / 2) * width, t, width, label=label)
    axes[0].set_xticks(np.arange(10), PATTERN_NAMES)
    axes[0].set_ylabel("share of pixels")
    axes[0].set_title("256 codes grouped into pattern types (number of neighbours >= centre)")
    axes[0].legend()

    im = axes[1].imshow(dist, cmap="viridis_r")
    short = [l.split(": ")[-1] for l in labels]
    axes[1].set_xticks(range(n), short, rotation=30)
    axes[1].set_yticks(range(n), short)
    for (i, j), v in np.ndenumerate(dist):
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v > dist.max() / 2 else "k")
    axes[1].set_title("Chi-square distance between\n256-bin histograms (0 = identical)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.savefig(OUT_DIR / "3_histogram_comparison.png", dpi=120)
    plt.close(fig)

    print("\nPattern-type shares (columns 0..9):")
    for label, t in zip(labels, types):
        print(f"  {label:24s} " + " ".join(f"{v:5.1%}" for v in t))
    print("\nChi-square distances:")
    for label, row in zip(labels, dist):
        print(f"  {label:24s} " + " ".join(f"{v:.3f}" for v in row))


def scene_classes(n_per_class=50, seed=0):
    """Does LBP separate scene types? Mean distance within vs between Intel classes."""
    rng = np.random.default_rng(seed)
    classes = sorted(p.name for p in INTEL_DIR.iterdir() if p.is_dir())
    hists = {}
    for c in classes:
        files = sorted((INTEL_DIR / c).glob("*.jpg"))
        picks = rng.choice(len(files), n_per_class, replace=False)
        hists[c] = np.array([lbp_histogram(lbp(load_gray(files[i]))) for i in picks])

    k = len(classes)
    dist = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            d = [chi_square(x, y) for x in hists[classes[a]] for y in hists[classes[b]]]
            if a == b:  # leave out each image compared with itself
                d = [v for v in d if v > 0]
            dist[a, b] = np.mean(d)

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), layout="constrained",
                             gridspec_kw=dict(width_ratios=[2, 1]))
    width = 0.8 / k
    for i, c in enumerate(classes):
        t = np.array([pattern_type_histogram(h) for h in hists[c]]).mean(axis=0)
        axes[0].bar(np.arange(10) + (i - (k - 1) / 2) * width, t, width, label=c)
    axes[0].set_xticks(np.arange(10), PATTERN_NAMES)
    axes[0].set_ylabel("mean share of pixels")
    axes[0].set_title(f"Intel scenes: mean pattern types ({n_per_class} images per class)")
    axes[0].legend()

    im = axes[1].imshow(dist, cmap="viridis_r")
    axes[1].set_xticks(range(k), classes, rotation=30)
    axes[1].set_yticks(range(k), classes)
    for (i, j), v in np.ndenumerate(dist):
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="w" if v > dist.max() * 0.9 else "k", fontweight="bold" if i == j else None)
    axes[1].set_title("Mean chi-square distance\n(diagonal = within class)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.savefig(OUT_DIR / "4_scene_classes.png", dpi=120)
    plt.close(fig)

    print("\nIntel classes, mean chi-square (row vs column):")
    print("  " + " " * 10 + " ".join(f"{c[:8]:>8s}" for c in classes))
    for c, row in zip(classes, dist):
        print(f"  {c:10s}" + " ".join(f"{v:8.3f}" for v in row))


if __name__ == "__main__":
    images = [(label, load_gray(CHOSEN_DIR / name)) for label, name in IMAGES]

    # Tasks 1-4: introduce the method using only the brick wall.
    label, img = images[0]
    plot_worked_example(img, label)
    example_results = plot_lbp_images(images[:1], filename="2_brick_lbp.png")
    plot_lbp_histograms(example_results, filename="4_brick_histogram.png")

    # Tasks 5-7: compare the same brick wall with three additional images.
    results = plot_lbp_images(images)
    plot_lbp_histograms(results)
    compare_histograms(results)
    scene_classes()

    print(f"\nFigures saved to {OUT_DIR.resolve()}")
