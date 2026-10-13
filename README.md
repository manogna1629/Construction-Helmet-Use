# Construction Helmet Use Recognition

A benchmarked, explainable CNN classification system that recognises whether a construction
worker's head is wearing a safety helmet. A custom CNN trained from scratch is compared against
ImageNet-pretrained MobileNetV2 and ResNet50, with Grad-CAM used to inspect what the models look at.

![Confusion matrices](results/figures/confusion_matrices.png)

## Results

Test set: 2,553 head crops. Positive class = `no_helmet` (the safety-critical case), threshold 0.5.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params | Latency (ms) |
|---|---|---|---|---|---|---|---|
| Custom CNN | 94.44% | 0.883 | 0.839 | 0.861 | 0.977 | 258,881 | 116.5 |
| MobileNetV2 | **97.34%** | **0.941** | **0.926** | **0.933** | **0.993** | 2,259,265 | 149.6 |
| ResNet50 | – | – | – | – | – | 23,589,761 | – |

- **MobileNetV2** is best on every quality metric. It misses 38 bare heads against the custom CNN's 84.
- **Custom CNN** is about 1/9 the size of MobileNetV2 and 22% faster, and was still improving when training stopped.
- **ResNet50** reached 97.50% validation accuracy after one epoch, but its run was interrupted in epoch 2, so it has no test results yet.

Latency is the mean single-image `predict()` time on CPU.

<p>
  <img src="results/figures/roc_curves.png" width="45%">
  <img src="results/figures/custom_cnn_history.png" width="53%">
</p>

## Dataset

[Hard Hat Workers](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) (Kaggle, andrewmvd):
5,000 site images with Pascal-VOC boxes for `helmet`, `head` and `person`.

| Step | Count |
|---|---|
| `helmet` + `head` boxes | 24,751 |
| Boxes under 20 px skipped | −7,673 |
| Crops written (10% padding) | 17,078 |
| Removed in cleaning (27 duplicates, 7 blurry) | −34 |
| **Final dataset** | **17,044** (13,410 helmet / 3,634 no_helmet) |

Split: 11,931 train / 2,560 validation / 2,553 test, at 160 × 160 px.
The dataset is downloaded automatically by the notebook and is not stored in this repo.

## Method

1. **Crop**: `voc_crop.py` converts every `helmet` box to class *helmet* and every `head` box to class *no_helmet*. `person` boxes are dropped.
2. **Clean**: `clean.py` flags near-duplicates (perceptual hash) and blurry crops (Laplacian variance < 15). It moves them to a review folder instead of deleting them.
3. **Load and augment**: `data.py` adds horizontal flips, small rotations, and brightness/contrast jitter.
4. **Models**: `models.py` defines the custom 4-block CNN, plus MobileNetV2 and ResNet50 with frozen ImageNet backbones and a new GAP → dropout → sigmoid head.
5. **Evaluate**: `evaluate.py` computes accuracy, precision, recall, F1, ROC-AUC, confusion matrices and latency.
6. **Explain**: `gradcam.py` produces Grad-CAM heat maps.

All models use binary cross-entropy, Adam (lr = 1e-3), batch size 32, early stopping on `val_loss`, and 2 epochs.

## Repository structure

```
Construction-Helmet-Use-Recognition/
├── notebooks/
│   └── Construction_Helmet_Use_Recognition.ipynb   full pipeline with outputs
├── src/
│   ├── voc_crop.py      VOC boxes → classification crops
│   ├── clean.py         duplicate / blur / corruption check
│   ├── data.py          tf.data loading, split, augmentation
│   ├── models.py        custom CNN + transfer-learning builders
│   ├── evaluate.py      metrics, ROC points, latency benchmark
│   └── gradcam.py       Grad-CAM for both model types
├── results/
│   ├── figures/         all plots from the notebook run
│   └── tables/          model_comparison.csv, class_balance.csv, training_log.txt
├── report/              IEEE-format report (main.tex, references.bib, figures/)
├── presentation/        slide deck (.pptx)
├── data/                created at runtime (git-ignored)
├── requirements.txt
└── LICENSE
```

## How to run (Google Colab)

1. Open the notebook in Colab, then go to **Runtime → Change runtime type → T4 GPU**.
2. Replace the first code cell with:
   ```python
   !git clone https://github.com/likithreddy77/Construction-Helmet-Use-Recognition.git
   %cd Construction-Helmet-Use-Recognition/notebooks
   !pip install -q -r ../requirements.txt
   ```
3. Run all cells. The notebook imports from `../src`, downloads the dataset with `kagglehub`, and writes to `../data` and `../results`.

To run locally, do the same from the `notebooks/` folder with Python 3.10+.

## Explainability

![Grad-CAM, custom CNN](results/figures/gradcam_custom_cnn_correct_prediction.png)

For the custom CNN, much of the Grad-CAM activation sits on the crop borders rather than the
helmet itself, which suggests it partly relies on padding and edge cues. Grad-CAM for the transfer
models currently fails with a `KeyError` (see Limitations).

## Limitations

- **Short CPU training**: the run had no GPU, so each model got only 2 epochs. The custom CNN had not converged.
- **ResNet50 not evaluated**: training was interrupted during epoch 2.
- **Unstable validation/test split**: the split uses `take()`/`skip()` on a dataset that reshuffles on every pass, so the two sets are not fixed and can overlap. You can see this in the test class counts, which differ between models (523 vs 514 no_helmet). Fix it by passing `shuffle=False` for the holdout, or by splitting the files beforehand.
- **Crop-level split**: crops from the same source image can land in both train and test.
- **No class weighting or threshold tuning**: the data is imbalanced 3.69 : 1.
- **Grad-CAM for transfer models**: Keras 3 does not connect the nested backbone layer to the outer model graph.

## Future work

- Train on a GPU for more epochs.
- Fine-tune the upper backbone layers.
- Split by source image.
- Add class weights and tune the decision threshold on validation data.
- Fix Grad-CAM for the transfer models.
- Pair the classifier with a head detector to monitor site video end to end.

## Report and slides

- `report/main.tex` is an IEEEtran paper with step-by-step working for every metric. Upload the `report/` folder to Overleaf to compile it.
- `presentation/Construction_Helmet_Use_Recognition.pptx` is a 13-slide deck with speaker notes.

## Author

**Iyallareddy Likith Reddy**, MSc Data Science, Arden University
GitHub: [@likithreddy77](https://github.com/likithreddy77)

## License

MIT, see [LICENSE](LICENSE). The dataset is subject to its own licence on Kaggle.
