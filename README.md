# Neural Network Experiments

This repository contains a set of neural-network experiments on MNIST-style digit classification. The project explores prediction analysis, custom handwritten digit generalization, learning curves, EarlyStopping, Dropout, L2 regularization, optimizer behavior, batch-size effects, and activation-function comparison.

## Project Overview

The main implementation is contained in `notebook.ipynb`. Each experiment saves its outputs inside the `results/` directory, while the written analysis for each task is stored inside the `submission/` directory.

The experiments focus on understanding how different training choices affect:

- convergence speed
- validation performance
- overfitting and underfitting
- training stability
- generalization to custom handwritten digits
- gradient behavior

## Repository Structure

```text
neural-network-experiments/
├── .gitignore
├── README.md
├── notebook.ipynb
├── data/
│   └── custom_digits/
│       ├── task02_original_digit.png
│       ├── task02_original_digit_2.png
│       └── task02_original_digit_3.png
├── results/
│   ├── activation_tests/
│   │   └── task09_activations/
│   │       ├── activation_gradient_norm_comparison.png
│   │       ├── activation_validation_loss_comparison.png
│   │       ├── gelu_loss.png
│   │       ├── relu_loss.png
│   │       ├── softsign_loss.png
│   │       ├── tanh_loss.png
│   │       └── task09_activation_results.txt
│   ├── batch_size_tests/
│   │   └── task08_batch_sizes/
│   │       ├── batch_8_loss.png
│   │       ├── batch_8_accuracy.png
│   │       ├── batch_32_loss.png
│   │       ├── batch_32_accuracy.png
│   │       ├── batch_128_loss.png
│   │       ├── batch_128_accuracy.png
│   │       ├── batch_size_validation_loss_comparison.png
│   │       └── task08_batch_size_results.txt
│   ├── loss_curves/
│   │   ├── task03_epochs/
│   │   │   ├── .gitkeep
│   │   │   ├── task03_5_epochs_loss.png
│   │   │   ├── task03_5_epochs_accuracy.png
│   │   │   ├── task03_10_epochs_loss.png
│   │   │   ├── task03_10_epochs_accuracy.png
│   │   │   ├── task03_20_epochs_loss.png
│   │   │   ├── task03_20_epochs_accuracy.png
│   │   │   └── task03_final_metrics.txt
│   │   ├── task04_early_stopping/
│   │   │   ├── task04_adam_patience_3_loss.png
│   │   │   ├── task04_adam_patience_3_accuracy.png
│   │   │   ├── task04_adam_patience_5_loss.png
│   │   │   ├── task04_adam_patience_5_accuracy.png
│   │   │   ├── task04_sgd_patience_3_loss.png
│   │   │   ├── task04_sgd_patience_3_accuracy.png
│   │   │   ├── task04_early_stopping_comparison.png
│   │   │   └── task04_early_stopping_results.txt
│   │   ├── task05_dropout/
│   │   │   ├── task05_dropout_0_0_loss.png
│   │   │   ├── task05_dropout_0_0_accuracy.png
│   │   │   ├── task05_dropout_0_1_loss.png
│   │   │   ├── task05_dropout_0_1_accuracy.png
│   │   │   ├── task05_dropout_0_3_loss.png
│   │   │   ├── task05_dropout_0_3_accuracy.png
│   │   │   └── task05_dropout_results.txt
│   │   └── task06_l2/
│   │       ├── task06_l2_0_0001_loss.png
│   │       ├── task06_l2_0_001_loss.png
│   │       ├── task06_l2_0_01_loss.png
│   │       └── task06_l2_results.txt
│   ├── optimizer_tests/
│   │   ├── .gitkeep
│   │   └── task07_optimizers/
│   │       ├── sgd_loss.png
│   │       ├── sgd_accuracy.png
│   │       ├── sgd-momentum_loss.png
│   │       ├── sgd-momentum_accuracy.png
│   │       ├── adam_loss.png
│   │       ├── adam_accuracy.png
│   │       ├── adamw_loss.png
│   │       ├── adamw_accuracy.png
│   │       └── task07_optimizer_results.txt
│   └── predictions/
│       ├── task1_prediction_results.txt
│       ├── task1_sample_666.png
│       ├── task1_sample_777.png
│       ├── task1_sample_888.png
│       ├── task02_prediction.txt
│       ├── task02_prediction_2.txt
│       ├── task02_prediction_3.txt
│       ├── task02_preprocessed_digit.png
│       ├── task02_preprocessed_digit_2.png
│       └── task02_preprocessed_digit_3.png
└── submission/
    ├── Task01_PredictionAnalysis.md
    ├── Task02_CustomDigit.md
    ├── Task03_Epochs.md
    ├── Task04_EarlyStopping.md
    ├── Task05_Dropout.md
    ├── Task06_L2.md
    ├── Task07_Optimizers.md
    ├── Task08_BatchSize.md
    └── Task09_Activations.md
```

## Tasks Summary

| Task | Topic | Main Purpose |
|---|---|---|
| Task 01 | Deep Prediction Analysis | Analyze model predictions on selected test samples. |
| Task 02 | Custom Image Generalization | Test the model on custom handwritten digits after preprocessing. |
| Task 03 | Epoch-Based Learning Curves | Compare training behavior across 5, 10, and 20 epochs. |
| Task 04 | EarlyStopping | Study how `patience` and optimizer choice affect stopping behavior. |
| Task 05 | Dropout | Compare different Dropout rates and their effect on overfitting. |
| Task 06 | L2 Regularization | Study how L2 strength affects weight magnitude and generalization. |
| Task 07 | Optimizers | Compare SGD, SGD with Momentum, Adam, and AdamW. |
| Task 08 | Batch Size | Analyze the effect of batch size on training speed, stability, and generalization. |
| Task 09 | Activation Functions | Compare ReLU, Tanh, Softsign, and GELU. |

## Results Overview

### Task 01 — Prediction Analysis

The model correctly classified the selected test samples:

| Sample Index | True Label | Predicted Label | Result |
|---:|---:|---:|---|
| 666 | 7 | 7 | Correct |
| 777 | 1 | 1 | Correct |
| 888 | 2 | 2 | Correct |

### Task 02 — Custom Digit Test

The custom digit experiments showed that the model can generalize to external handwritten inputs, but confidence depends on how closely the input matches the MNIST training distribution.

| Custom Digit | Predicted Label | Confidence |
|---:|---:|---:|
| 8 | 8 | 0.5193 |
| 4 | 4 | 0.8963 |
| 3 | 3 | 1.0000 |

### Task 03 — Epoch Experiments

Increasing epochs improved training performance, but validation behavior showed overfitting after the best validation region.

| Epochs | Final Train Loss | Final Val Loss | Final Train Accuracy | Final Val Accuracy |
|---:|---:|---:|---:|---:|
| 5 | 0.0714 | 0.0963 | 0.9783 | 0.9734 |
| 10 | 0.0308 | 0.0862 | 0.9908 | 0.9774 |
| 20 | 0.0103 | 0.1340 | 0.9970 | 0.9748 |

### Task 04 — EarlyStopping

Adam triggered EarlyStopping quickly because validation loss stopped improving early, while SGD continued improving until the maximum epoch limit.

| Experiment | Best Epoch | Stopped Epoch | Best Val Loss | EarlyStopping Triggered |
|---|---:|---:|---:|---|
| Adam — Patience 3 | 4 | 7 | 0.0986 | Yes |
| Adam — Patience 5 | 4 | 9 | 0.0986 | Yes |
| SGD — Patience 3 | 50 | 50 | 0.0836 | No |

### Task 05 — Dropout

Dropout reduced overfitting by making the model less dependent on fixed neuron combinations.

| Dropout Rate | Final Train Loss | Final Val Loss | Best Val Loss | Final Val Accuracy |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0089 | 0.1296 | 0.0979 | 0.9736 |
| 0.1 | 0.0337 | 0.0949 | 0.0750 | 0.9792 |
| 0.3 | 0.0886 | 0.0748 | 0.0748 | 0.9796 |

### Task 06 — L2 Regularization

L2 regularization reduced weight magnitude as the penalty increased. Very strong L2 caused underfitting.

| L2 Value | Final Val Loss | Final Val Accuracy | Weight L2 Norm |
|---:|---:|---:|---:|
| 0.0001 | 0.1200 | 0.9760 | 18.2197 |
| 0.001 | 0.1579 | 0.9722 | 7.8453 |
| 0.01 | 0.2347 | 0.9636 | 3.2773 |

### Task 07 — Optimizer Comparison

Adam and AdamW converged fastest, while SGD with Momentum achieved the best validation loss in this experiment.

| Optimizer | Best Val Loss | Best Epoch | Convergence Epoch | Final Val Accuracy |
|---|---:|---:|---:|---:|
| SGD | 0.1101 | 20 | 20 | 0.9694 |
| SGD with Momentum | 0.0814 | 10 | 9 | 0.9744 |
| Adam | 0.0986 | 4 | 4 | 0.9714 |
| AdamW | 0.0985 | 4 | 4 | 0.9736 |

### Task 08 — Batch Size

Small batch size produced more updates and more unstable validation behavior. Batch size 128 trained fastest and achieved the lowest best validation loss.

| Batch Size | Steps per Epoch | Best Val Loss | Final Val Accuracy | Training Time |
|---:|---:|---:|---:|---:|
| 8 | 6875 | 0.0966 | 0.9744 | 517.55 s |
| 32 | 1719 | 0.0979 | 0.9748 | 185.63 s |
| 128 | 430 | 0.0905 | 0.9718 | 65.68 s |

### Task 09 — Activation Functions

Softsign achieved the best validation loss, while GELU achieved the highest final validation accuracy.

| Activation | Best Val Loss | Final Val Loss | Final Val Accuracy | Hidden Gradient Norm |
|---|---:|---:|---:|---:|
| ReLU | 0.0979 | 0.1296 | 0.9736 | 0.277891 |
| Tanh | 0.0953 | 0.1157 | 0.9748 | 0.373865 |
| Softsign | 0.0890 | 0.0994 | 0.9746 | 0.246544 |
| GELU | 0.0964 | 0.1386 | 0.9754 | 0.200162 |

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow
- Google Colab / Jupyter Notebook

## How to Run

1. Open `notebook.ipynb` in Jupyter Notebook or Google Colab.
2. Run the notebook cells in order.
3. Generated plots and text results will be saved automatically under `results/`.
4. Written task reports are available under `submission/`.

## Notes

- The repository currently contains documented reports for Task 01 through Task 09.
- The `results/` directory stores both visual plots and `.txt` summaries for reproducibility.
- The `data/custom_digits/` directory stores the original custom handwritten digit images used in Task 02.
