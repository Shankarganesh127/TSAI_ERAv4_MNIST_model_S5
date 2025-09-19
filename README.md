# 🧠 ERAv4 MNIST Model S5

A compact, high-accuracy CNN for MNIST digit classification.  
**Parameter count:** < 20,000 | **Best accuracy:** >99.4% in 20 epochs

---

## 📦 Project Files

| File Name                        | Description                                 |
|----------------------------------|---------------------------------------------|
| `ERAv4_MNIST_model_S5.ipynb`     | Final model notebook                        |
| `main.py`                        | Python script version                       |
| `ERAv4_MNIST_model_experiments.ipynb` | Experiments notebook                  |
| `pyproject.toml`                 | Python requirements                         |

---

## 🏗️ Model Architecture

<details>
<summary>Layer-wise Parameter Details</summary>

| Layer (type)         | Params | Details |
|----------------------|-------:|---------|
| **conv1.0 (Conv2d)** |     36 | 1→4 channels, kernel 3x3, no bias |
| **conv1.1 (BatchNorm2d)** | 8 | 4 features, affine |
| **conv1.3 (Conv2d)** |    288 | 4→8 channels, kernel 3x3, no bias |
| **conv1.4 (BatchNorm2d)** | 16 | 8 features, affine |
| **conv1.6 (MaxPool2d)** | 0 | kernel 2, stride 2 |
| **conv2.0 (Conv2d)** | 1152 | 8→16 channels, kernel 3x3, no bias |
| **conv2.1 (BatchNorm2d)** | 32 | 16 features, affine |
| **conv2.3 (Conv2d)** | 4608 | 16→32 channels, kernel 3x3, no bias |
| **conv2.4 (BatchNorm2d)** | 64 | 32 features, affine |
| **conv2.6 (MaxPool2d)** | 0 | kernel 2, stride 2 |
| **conv3.0 (Conv2d)** | 11520 | 32→40 channels, kernel 3x3, no bias |
| **conv3.1 (BatchNorm2d)** | 80 | 40 features, affine |
| **gap (AdaptiveAvgPool2d)** | 0 | output size 1 |
| **fc (Linear)** | 410 | 40→10, bias |
| **Total** | **18214** |  |

</details>

---

## 🔎 Summary

- **BatchNorm2d layers used:** 5
- **Dropout layers used:** 0
- **Fully Connected (Linear) layers used:** 1
- **Global Average Pooling layers used:** 1

---

## 🚀 Training Log (Best Results)

| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 5     | 98.31%    | 99.20%   |
| 10    | 98.92%    | 99.40%   |
| 15    | 98.97%    | 99.39%   |
| 20    | 99.03%    | 99.37%   |

<details>
<summary>Full Training Log</summary>

```
Test set: Average loss: 0.0001, Accuracy: 9802/10000 (98.02%)

Epoch 2
Train Loss=0.0082 Accuracy=97.35: 100%|██████████| 938/938 [01:19<00:00, 11.83it/s]

Test set: Average loss: 0.0001, Accuracy: 9807/10000 (98.07%)

Epoch 3
Train Loss=0.0154 Accuracy=97.87: 100%|██████████| 938/938 [01:18<00:00, 11.93it/s]

Test set: Average loss: 0.0000, Accuracy: 9889/10000 (98.89%)

Epoch 4
Train Loss=0.0138 Accuracy=98.14: 100%|██████████| 938/938 [01:21<00:00, 11.44it/s]

Test set: Average loss: 0.0000, Accuracy: 9829/10000 (98.29%)

Epoch 5
Train Loss=0.1234 Accuracy=98.31: 100%|██████████| 938/938 [01:18<00:00, 12.02it/s]

Test set: Average loss: 0.0000, Accuracy: 9920/10000 (99.20%)

Epoch 6
Train Loss=0.1804 Accuracy=98.44: 100%|██████████| 938/938 [01:28<00:00, 10.61it/s]

Test set: Average loss: 0.0000, Accuracy: 9884/10000 (98.84%)

Epoch 7
Train Loss=0.1565 Accuracy=98.87: 100%|██████████| 938/938 [01:30<00:00, 10.37it/s]

Test set: Average loss: 0.0000, Accuracy: 9933/10000 (99.33%)

Epoch 8
Train Loss=0.0069 Accuracy=98.91: 100%|██████████| 938/938 [01:22<00:00, 11.42it/s]

Test set: Average loss: 0.0000, Accuracy: 9937/10000 (99.37%)

Epoch 9
Train Loss=0.0294 Accuracy=98.94: 100%|██████████| 938/938 [01:19<00:00, 11.76it/s]

Test set: Average loss: 0.0000, Accuracy: 9935/10000 (99.35%)

Epoch 10
Train Loss=0.0284 Accuracy=98.92: 100%|██████████| 938/938 [01:16<00:00, 12.25it/s]

Test set: Average loss: 0.0000, Accuracy: 9940/10000 (99.40%)

Epoch 11
Train Loss=0.0027 Accuracy=98.93: 100%|██████████| 938/938 [01:18<00:00, 12.00it/s]

Test set: Average loss: 0.0000, Accuracy: 9940/10000 (99.40%)

Epoch 12
Train Loss=0.0756 Accuracy=98.98: 100%|██████████| 938/938 [01:19<00:00, 11.77it/s]

Test set: Average loss: 0.0000, Accuracy: 9943/10000 (99.43%)

Epoch 13
Train Loss=0.0169 Accuracy=98.98: 100%|██████████| 938/938 [01:20<00:00, 11.64it/s]

Test set: Average loss: 0.0000, Accuracy: 9942/10000 (99.42%)

Epoch 14
Train Loss=0.0050 Accuracy=99.00: 100%|██████████| 938/938 [01:33<00:00, 10.03it/s]

Test set: Average loss: 0.0000, Accuracy: 9940/10000 (99.40%)

Epoch 15
Train Loss=0.0357 Accuracy=98.97: 100%|██████████| 938/938 [01:21<00:00, 11.51it/s]

Test set: Average loss: 0.0000, Accuracy: 9939/10000 (99.39%)

Epoch 16
Train Loss=0.0110 Accuracy=98.97: 100%|██████████| 938/938 [01:25<00:00, 11.01it/s]

Test set: Average loss: 0.0000, Accuracy: 9940/10000 (99.40%)

Epoch 17
Train Loss=0.0844 Accuracy=99.00: 100%|██████████| 938/938 [01:27<00:00, 10.71it/s]

Test set: Average loss: 0.0000, Accuracy: 9941/10000 (99.41%)

Epoch 18
Train Loss=0.0256 Accuracy=99.01: 100%|██████████| 938/938 [01:20<00:00, 11.71it/s]

Test set: Average loss: 0.0000, Accuracy: 9939/10000 (99.39%)

Epoch 19
Train Loss=0.0460 Accuracy=99.02: 100%|██████████| 938/938 [01:19<00:00, 11.76it/s]

Test set: Average loss: 0.0000, Accuracy: 9944/10000 (99.44%)

Epoch 20
Train Loss=0.1812 Accuracy=99.03: 100%|██████████| 938/938 [01:41<00:00,  9.25it/s]

Test set: Average loss: 0.0000, Accuracy: 9937/10000 (99.37%)
```
</details>

---

## 📝 How to Run

```bash
# Install requirements
pip install -r requirements.txt

# Run notebook
jupyter notebook ERAv4_MNIST_model_S5.ipynb

# Or run as script
python main.py
```

---

## 📊 Results

- Achieved **>99.4%** test accuracy in 20 epochs
- Model size: **18,214 parameters**

---

## 📧 Contact

For questions, reach out via [GitHub Issues](https://github.com/Shankarganesh127/TSAI_ERAv4_MNIST_model_S5/issues).

---