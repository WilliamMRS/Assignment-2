# Blob detection experiments

This folder contains the implementation and experimental outputs for the blob
detection task. The report has not been written yet.

From the repository root, run:

```sh
MPLCONFIGDIR=/tmp/assignment2-mpl image_processing/venv/bin/python image_processing/blob/blob_detection.py
MPLCONFIGDIR=/tmp/assignment2-mpl image_processing/venv/bin/python -m unittest discover -s image_processing/blob -p 'test_*.py' -v
```

The script uses NumPy, SciPy, scikit-image, and Matplotlib from the project's
existing environment. Paths are resolved relative to the script. It can also
process other RGB/RGBA heatmaps with `--images path1.png path2.png`, and save to a
different directory with `--output directory`.

## Input selection

Four original-resolution (960 x 540) images from the provided `heatmap-nn`
dataset are copied byte-for-byte into `chosen_images/`. These are a small,
deliberately selected set illustrating different detection behaviors, rather
than a random sample or an evaluation of the entire dataset.

| Saved image | Original dataset path | Reason for selection |
| --- | --- | --- |
| `hm1_1.png` | `data/heatmap-nn/hm1/Heat Map (1).png` | Large connected region, isolated regions, clipped corner region |
| `hm20_27.png` | `data/heatmap-nn/hm20/Heat Map (27).png` | Small isolated regions and an elongated region along the bottom border |
| `hm20_9.png` | `data/heatmap-nn/hm20/Heat Map (9).png` | Nearby hot regions connected by weaker colors |
| `hm50_28.png` | `data/heatmap-nn/hm50/Heat Map (28).png` | Separated regions of several sizes, including weaker warm patches |

## Algorithm and measurement definitions

1. Convert RGB channels to floating point in [0, 1]. Composite transparent
   pixels over white for RGBA images.
2. Compute a warm-color score: `clip(R - (G + B) / 2, 0, 1)`. Pure red scores
   1; yellow scores 0.5; neutral gray, white, black, blue, and green score 0.
3. Smooth the score with a Gaussian filter (default sigma = 1.5 pixels,
   reflected image boundaries).
4. Threshold the smoothed score at 0.35.
5. Label 8-connected foreground components. Keep components with area at
   least 80 pixels. Each retained component is one blob.
6. Measure the number of foreground pixels (area), the unweighted centroid,
   bounding box, and equivalent-area radius `sqrt(area / pi)` for each blob.

Coordinates are zero-based: x increases rightwards and y increases downwards.
Bounding box starts are inclusive and ends are exclusive. Circle centers are
centroids, and their areas equal the measured component areas. The circles do
not describe the actual boundaries of elongated or irregular components.
Border-touching blobs are retained and flagged; their measured areas include
only the visible part. No cropping or resizing is applied.

The score selects hot colors from a rendered overlay. It does not reconstruct
the original heatmap values. Green and blue patches are excluded by design.
Colored text or graphics can also pass the rule, and overlapping hot regions
can become one component. There are no supplied ground-truth blob annotations
in this experiment, so parameter sensitivity and visual inspection do not
establish precision, recall, or an optimal parameter setting. Region counts
must not be interpreted as counts of people, fixations, or underlying events.

## Outputs

- `output/1_pipeline.png`: original, score, retained mask, detections.
- `output/2_detections.png`: original and annotated version of every input.
- `output/3_statistics.png`: blob IDs, areas, centers, radii, and border flags.
- `output/4_parameter_sweep.png`: count and coverage for every image.
- `output/5_parameter_examples.png`: visual threshold, smoothing, and area-filter comparisons.
- `output/image_summary.csv`: per-image counts, areas, coverage, border counts.
- `output/blob_statistics.csv`: every blob's size, position, and bounding box.
- `output/parameter_sweep.csv`: all 36 image/parameter experiments.
- `output/parameters.json`: baseline settings, input paths, measurement conventions.

Each sweep varies one parameter while keeping the other two at baseline:
threshold = 0.20/0.35/0.60; smoothing sigma = 0/1.5/4 pixels; minimum area =
80/300/1500 pixels. The baseline is a working setting for illustration, not a
validated optimum. Keep these same images for the later contour task, as the
assignment requires a comparison on the same dataset.
