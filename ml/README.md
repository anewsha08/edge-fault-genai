# ml/

Dataset preparation, training, quantization, and model export.

## Plan
1. `prepare_data.py`: load public vibration data (e.g., CWRU), resample to match the MCU rate, cut into 256-sample windows, label them.
2. `train.py`: small 1D CNN or MLP on FFT features.
3. `quantize.py`: int8 post-training quantization with a representative dataset.
4. `export.py`: convert to a C array for TFLite Micro.
5. `evaluate.py`: confusion matrix; accuracy float vs int8.

## Important
- Split train/test by recording or machine condition, not by random window, otherwise accuracy is inflated by leakage.
- Match the sampling rate and units between the dataset and your sensor, or the model will not transfer to real hardware.
- Large data goes in `ml/data/` (git-ignored).
