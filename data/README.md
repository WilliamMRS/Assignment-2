# Data

Datasets are not committed. Download them and place them here so the layout is:

```
data/
├── vehicles/                       # one subfolder per class (sedan/, pickup/, motorcycle/, ...) of .jpg images
├── facial-emotion-recognition/
├── heatmap-nn/
└── intel-image-classification (1)/ # seg_train/, seg_test/, seg_pred/
```

`3_hog_features.py` needs `data/vehicles/` (e.g. `data/vehicles/sedan/PIC_1.jpg`).
