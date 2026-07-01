# Task 07 — Optimizer Comparison Challenge

## 1. Objective

The objective of this experiment is to compare four optimizers using the same neural-network architecture:

- SGD
- SGD with Momentum
- Adam
- AdamW

The comparison focuses on:

- Convergence speed.
- Training stability.
- Validation performance.
- Overfitting behavior.
- How each optimizer navigates the loss landscape.

All models used the same dataset, architecture, random seed, batch size, and number of epochs.

---

## 2. Optimizer Configurations

```python
SGD:
keras.optimizers.SGD(
    learning_rate=0.01
)

SGD with Momentum:
keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9
)

Adam:
keras.optimizers.Adam(
    learning_rate=0.001
)

AdamW:
keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.0001
)
```

Each model was trained for `20` epochs using a batch size of `32`.

---

## 3. Results

| Optimizer | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc | Best Val Loss | Best Loss Epoch | Best Val Acc | Best Acc Epoch | Convergence Epoch | Final Val-Loss STD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD | 0.1245 | 0.1100 | 96.50% | 96.94% | 0.1100 | 20 | 96.94% | 20 | 20 | 0.003850 |
| SGD with Momentum | 0.0147 | 0.0893 | 99.80% | 97.32% | 0.0816 | 10 | 97.56% | 8 | 8 | 0.000724 |
| Adam | 0.0083 | 0.1304 | 99.79% | 97.52% | 0.0986 | 4 | 97.52% | 20 | 4 | 0.001615 |
| AdamW | 0.0078 | 0.1401 | 99.83% | 97.28% | 0.0995 | 5 | 97.54% | 19 | 4 | 0.007796 |

Training times were similar:

| Optimizer | Training Time |
|---|---:|
| SGD | 97.94 seconds |
| SGD with Momentum | 95.27 seconds |
| Adam | 106.35 seconds |
| AdamW | 105.89 seconds |

The small time differences may be influenced by the Colab runtime and should not be treated as precise optimizer benchmarks.

---

## 4. Loss and Accuracy Curves

### SGD

![SGD Loss Curves](../results/optimizer_tests/task07_optimizers/sgd_loss.png)

![SGD Accuracy Curves](../results/optimizer_tests/task07_optimizers/sgd_accuracy.png)

### SGD with Momentum

![SGD Momentum Loss Curves](../results/optimizer_tests/task07_optimizers/sgd-momentum_loss.png)

![SGD Momentum Accuracy Curves](../results/optimizer_tests/task07_optimizers/sgd-momentum_accuracy.png)

### Adam

![Adam Loss Curves](../results/optimizer_tests/task07_optimizers/adam_loss.png)

![Adam Accuracy Curves](../results/optimizer_tests/task07_optimizers/adam_accuracy.png)

### AdamW

![AdamW Loss Curves](../results/optimizer_tests/task07_optimizers/adamw_loss.png)

![AdamW Accuracy Curves](../results/optimizer_tests/task07_optimizers/adamw_accuracy.png)

---

## 5. Short Analysis

### SGD — Slow but Steady Convergence

SGD showed the slowest convergence among the four optimizers.

Its training and validation losses decreased continuously throughout all `20` epochs. The best validation loss and validation accuracy were both reached at epoch `20`, indicating that the model was still improving when training ended.

The final validation accuracy was `96.94%`, which was the lowest among the tested optimizers. However, the curves did not show clear overfitting.

The final validation loss was slightly lower than the training loss:

```text
0.1100 - 0.1245 = -0.0145
```

This can occur because the training loss is averaged across batches while the weights are being updated. Validation loss is calculated after the epoch using the final updated weights.

The results suggest that SGD may have benefited from additional epochs or learning-rate scheduling.

---

### SGD with Momentum — Best Overall Validation Loss

Adding Momentum significantly accelerated SGD.

The model reached its best validation accuracy of `97.56%` at epoch `8` and its best validation loss of `0.0816` at epoch `10`.

These were the best validation results among the four optimizers.

Momentum allowed the optimizer to preserve useful update directions across batches, helping it move faster through consistent gradients and reducing unnecessary oscillation.

After approximately epoch `10`, training loss continued decreasing while validation loss gradually increased. This indicates that overfitting began after the model reached its best validation performance.

Its final five-epoch validation-loss standard deviation was only `0.000724`, the lowest of the four experiments. This indicates that the validation curve changed very little near the end, although it was slowly moving upward.

---

### Adam — Fast Convergence but Strong Overfitting

Adam converged very quickly and reached its best validation loss of `0.0986` at epoch `4`.

Its final validation accuracy was `97.52%`, the highest final-epoch validation accuracy among the four models.

However, after the early epochs, training loss continued decreasing toward zero while validation loss increased to `0.1304`.

The final loss gap was:

```text
0.1304 - 0.0083 = 0.1221
```

This large gap indicates strong overfitting.

Adam was highly effective at minimizing the training loss, but the absence of EarlyStopping or sufficient regularization allowed it to continue fitting training-specific patterns after validation performance had stopped improving.

---

### AdamW — Fast Convergence with Weight Decay

AdamW also converged quickly, reaching the convergence threshold at epoch `4` and its best validation loss of `0.0995` at epoch `5`.

AdamW applies weight decay separately from the adaptive gradient update. This is intended to control weight growth more consistently than applying a traditional L2 penalty inside Adam.

However, in this experiment, the selected weight-decay value of `0.0001` was not strong enough to prevent late overfitting.

Its training loss decreased to `0.0078`, while validation loss increased to `0.1401`, producing the largest final loss gap:

```text
0.1401 - 0.0078 = 0.1323
```

The final five-epoch validation-loss standard deviation was `0.007796`, the highest among the four optimizers, reflecting larger fluctuations near the end of training.

AdamW did not outperform Adam under these specific settings. Its performance could change with a different weight-decay value, learning rate, or training duration.

---

## 6. Convergence Speed Comparison

According to the convergence metric, the optimizers ranked as follows:

| Rank | Optimizer | Convergence Epoch |
|---:|---|---:|
| 1 | Adam | 4 |
| 1 | AdamW | 4 |
| 3 | SGD with Momentum | 8 |
| 4 | SGD | 20 |

Adam and AdamW converged fastest because they adapted the learning rate separately for each parameter.

SGD with Momentum was slower than the adaptive optimizers but substantially faster than standard SGD.

Standard SGD required the full 20 epochs and was still improving at the end of training.

---

## 7. Stability Comparison

Using the standard deviation of the final five validation-loss values:

| Optimizer | Final Five-Epoch Val-Loss STD |
|---|---:|
| SGD with Momentum | 0.000724 |
| Adam | 0.001615 |
| SGD | 0.003850 |
| AdamW | 0.007796 |

SGD with Momentum produced the least fluctuating validation loss near the end.

AdamW produced the largest late-stage fluctuations.

However, this standard deviation does not measure instability alone. For example, SGD had a higher value partly because its validation loss was still consistently decreasing rather than oscillating around a fixed value.

---

## 8. How Each Optimizer Navigates the Loss Landscape

### SGD

SGD updates the parameters using the current batch gradient:

```text
Current gradient
→ One update direction
→ Same global learning rate for all parameters
```

It follows the local slope directly and can therefore move slowly through flat regions or oscillate across narrow valleys.

Its simplicity may sometimes help it reach solutions that generalize well, but it often requires careful learning-rate tuning and more epochs.

### SGD with Momentum

Momentum combines the current gradient with a moving accumulation of previous update directions.

```text
Current gradient
+
Previous direction
=
Smoothed update
```

It accelerates movement when gradients repeatedly point in the same direction and reduces side-to-side oscillation.

This allowed it to converge much faster than standard SGD and achieve the best validation loss in this experiment.

### Adam

Adam maintains two moving estimates:

- The average direction of the gradients.
- The average squared magnitude of the gradients.

It then assigns an adaptive update size to each parameter.

Parameters with consistently large or unstable gradients receive relatively smaller updates, while parameters with smaller gradients may receive relatively larger updates.

This allows Adam to move efficiently through loss landscapes containing:

- Different gradient scales.
- Flat regions.
- Noisy gradients.
- Narrow valleys.

### AdamW

AdamW uses Adam’s adaptive updates but applies weight decay separately.

```text
Adaptive gradient update
+
Independent weight decay
```

This separation is intended to make regularization more predictable. However, the weight-decay coefficient must still be tuned appropriately.

---

## 9. Why Adam Often Outperforms Classical Optimizers

Adam often performs well because it combines two useful ideas:

1. **Momentum-like gradient smoothing**

   Adam tracks a moving average of previous gradients, reducing the effect of noisy batch updates.

2. **Adaptive per-parameter learning rates**

   Each weight receives an update size based on its own gradient history rather than using the same effective step size for every parameter.

This allows Adam to:

- Converge quickly.
- Require less manual learning-rate tuning.
- Handle differently scaled gradients.
- Perform well with noisy or sparse gradients.

However, Adam does not always produce the best generalization.

In this experiment, Adam converged faster than SGD with Momentum, but SGD with Momentum achieved a lower best validation loss and a slightly higher best validation accuracy.

Therefore, Adam’s main advantage was optimization speed, while Momentum achieved the best validation performance.

---

## 10. Key Takeaway

Standard SGD converged slowly but continued improving throughout all 20 epochs.

SGD with Momentum provided the best overall validation performance, achieving the lowest best validation loss of `0.0816` and the highest best validation accuracy of `97.56%`.

Adam and AdamW converged the fastest, but both began overfitting after only a few epochs.

Adam achieved the highest final validation accuracy, while AdamW did not provide an advantage under the selected weight-decay setting.

The experiment demonstrates that faster optimization does not always produce better generalization. In this case, SGD with Momentum provided the best balance between convergence speed, stability, and validation performance.
