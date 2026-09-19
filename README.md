# Bear Classification with timm + NLLLoss (5-class)

This notebook trains a timm backbone (default: `resnet18`) on a single ImageFolder dataset root, does a reproducible 80/20 split, and reports:
- accuracy
- macro-F1
- per-class accuracy
- confusion matrix

It also uses:
- LR scheduling (`ReduceLROnPlateau`)
- early stopping
- optional class balancing with `WeightedRandomSampler
`
