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

## 2. Code Used

```python
def create_optimizer(optimizer_name):
    # Return a new optimizer instance for each experiment.
    optimizers = {
        "SGD":          keras.optimizers.SGD(learning_rate=0.01),
        "SGD_Momentum": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "Adam":         keras.optimizers.Adam(),
        "AdamW":        keras.optimizers.AdamW(weight_decay=0.0001)
    }
    return optimizers[optimizer_name]


def create_optimizer_model(optimizer_name, seed=42):
    # Create a fresh model using the selected optimizer.
    # Every experiment uses the same architecture and seed.

    # Reset the random seed so models start with the same initial weights.
    keras.utils.set_random_seed(seed)

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=create_optimizer(optimizer_name),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_curves(history, optimizer_name, metric):
    # Plot and save training vs validation curves for a given metric.
    display_name = optimizer_name.replace("_", " with ")
    file_name = optimizer_name.lower().replace("_", "-")
    epoch_range = range(1, len(history.history[metric]) + 1)

    plt.figure(figsize=(7, 5))

    plt.plot(
        epoch_range,
        history.history[metric],
        label=f"Training {metric.capitalize()}"
    )

    plt.plot(
        epoch_range,
        history.history[f"val_{metric}"],
        label=f"Validation {metric.capitalize()}"
    )

    plt.title(f"{metric.capitalize()} Curves — {display_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid()

    output_path = task7_results_dir / f"{file_name}_{metric}.png"

    # Save the figure before displaying it.
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved: {output_path}")


# Train all four optimizer configurations.
optimizer_names = ["SGD", "SGD_Momentum", "Adam", "AdamW"]
optimizer_results = {}

for optimizer_name in optimizer_names:

    # Create a fresh model for this optimizer.
    model = create_optimizer_model(optimizer_name, seed=42)

    # Record the training time.
    start_time = time.time()

    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(x_val, y_val),
        shuffle=True,
        verbose=1
    )

    training_time = time.time() - start_time

    # Plot and save loss and accuracy curves.
    plot_curves(history, optimizer_name, "loss")
    plot_curves(history, optimizer_name, "accuracy")

    # Find the lowest validation loss.
    best_val_loss = np.min(history.history["val_loss"])

    # Find the epoch that achieved the lowest validation loss.
    best_epoch = np.argmin(history.history["val_loss"]) + 1

    # Estimate convergence speed:
    # First epoch whose validation loss is within 1% of the best
    # validation loss.
    convergence_epoch = next(
        i + 1
        for i, value in enumerate(history.history["val_loss"])
        if value <= best_val_loss * 1.01
    )

    # Measure validation-loss stability over the final five epochs.
    validation_loss_std = np.std(
        history.history["val_loss"][-5:]
    )

    # Store the results needed for comparison.
    optimizer_results[optimizer_name] = {
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "final_train_accuracy": history.history["accuracy"][-1],
        "final_val_accuracy": history.history["val_accuracy"][-1],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "convergence_epoch": convergence_epoch,
        "validation_loss_std": validation_loss_std,
        "training_time": training_time
    }
```

Each model was trained for `20` epochs using a batch size of `32`.

The learning rates were not identical because the optimizers use different update mechanisms. SGD commonly starts with a larger learning rate, while Adam and AdamW commonly use a smaller value because they adapt the effective update size for each parameter.

---

## 3. Results

| Optimizer | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc | Best Val Loss | Best Loss Epoch | Convergence Epoch | Final Val-Loss STD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD | 0.1245 | 0.1101 | 96.50% | 96.94% | 0.1101 | 20 | 20 | 0.003842 |
| SGD with Momentum | 0.0147 | 0.0891 | 99.80% | 97.44% | 0.0814 | 10 | 9 | 0.000657 |
| Adam | 0.0083 | 0.1530 | 99.79% | 97.14% | 0.0986 | 4 | 4 | 0.009132 |
| AdamW | 0.0091 | 0.1370 | 99.77% | 97.36% | 0.0985 | 4 | 4 | 0.003487 |

Training times were similar:

| Optimizer | Training Time |
|---|---:|
| SGD | 109.03 seconds |
| SGD with Momentum | 96.97 seconds |
| Adam | 102.46 seconds |
| AdamW | 104.68 seconds |

The small time differences may be influenced by the Colab runtime and should not be treated as precise optimizer benchmarks.

---

## 4. Loss and Accuracy Curves

### SGD

<table>
  <tr>
    <th>Loss Curves</th>
    <th>Accuracy Curves</th>
  </tr>
  <tr>
    <td>
      <img src="../results/optimizer_tests/task07_optimizers/sgd_loss.png" width="450">
    </td>
    <td>
      <img src="../results/optimizer_tests/task07_optimizers/sgd_accuracy.png" width="450">
    </td>
  </tr>
</table>

---

### SGD with Momentum

<table>
  <tr>
    <th>Loss Curves</th>
    <th>Accuracy Curves</th>
  </tr>
  <tr>
    <td>
      <img src="../results/optimizer_tests/task07_optimizers/sgd-momentum_loss.png" width="450">
    </td>
    <td>
      <img src="../results/optimizer_tests/task07_optimizers/sgd-momentum_accuracy.png" width="450">
    </td>
  </tr>
</table>

---

### Adam

<table>
  <tr>
    <th>Loss Curves</th>
    <th>Accuracy Curves</th>
  </tr>
  <tr>
    <td>
      <img src="../results/optimizer_tests/task07_optimizers/adam_loss.png" width="450">
    </td>
    <td>
      <img src="../results/optimizer_tests/task07_optimizers/adam_accuracy.png" width="450">
    </td>
  </tr>
</table>

---

### AdamW

<table>
  <tr>
    <th>Loss Curves</th>
    <th>Accuracy Curves</th>
  </tr>
  <tr>
    <td>
      <img src="../results/optimizer_tests/task07_optimizers/adamw_loss.png" width="450">
    </td>
    <td>
      <img src="../results/optimizer_tests/task07_optimizers/adamw_accuracy.png" width="450">
    </td>
  </tr>
</table>

---
## 5. Short Analysis

### SGD — Slow but Still Improving

SGD had the slowest convergence among the four optimizers.

Its training and validation losses continued decreasing until the final epoch. The best validation loss was reached at Epoch `20`, which means the model was still improving when training stopped.

SGD did not show clear overfitting, but it also had the lowest final validation accuracy: `96.94%`.

This suggests that standard SGD may need more epochs or a learning-rate schedule to reach stronger performance.

---

### SGD with Momentum — Best Validation Performance

SGD with Momentum improved much faster than standard SGD.

It reached the convergence threshold at Epoch `9` and achieved the best validation loss at Epoch `10`.

This optimizer achieved the strongest validation performance in this experiment:

```text
Best Validation Loss: 0.0814
Final Validation Loss: 0.0891
Final Validation Accuracy: 97.44%
```

Momentum helped by preserving useful update directions across batches. This made optimization faster and reduced unnecessary oscillation compared with standard SGD.

After the best epoch, training loss continued decreasing while validation loss slightly increased, indicating mild overfitting.

---

### Adam — Fastest Convergence, Strongest Overfitting

Adam converged very quickly and reached its best validation loss at Epoch `4`.

However, after that point, training loss continued decreasing while validation loss increased to `0.1530`.

The final loss gap was the largest among the four optimizers:

```text
0.1530 - 0.0083 = 0.1447
```

This indicates strong overfitting.

Adam was very effective at minimizing training loss, but without EarlyStopping or stronger regularization, it continued fitting training-specific patterns after validation performance stopped improving.

---

### AdamW — Fast Convergence with Better Regularization than Adam

AdamW also converged quickly and reached its best validation loss at Epoch `4`.

Its best validation loss was almost identical to Adam:

```text
Adam Best Val Loss:  0.0986
AdamW Best Val Loss: 0.0985
```

However, AdamW had a lower final validation loss than Adam:

```text
Adam Final Val Loss:  0.1530
AdamW Final Val Loss: 0.1370
```

This suggests that AdamW's weight decay helped reduce late overfitting compared with Adam, although it did not eliminate overfitting completely.

---

## 6. Comparison Summary

### Convergence Speed

| Rank | Optimizer | Convergence Epoch |
|---:|---|---:|
| 1 | Adam | 4 |
| 1 | AdamW | 4 |
| 3 | SGD with Momentum | 9 |
| 4 | SGD | 20 |

Adam and AdamW converged the fastest because they use adaptive per-parameter update sizes.

SGD with Momentum was slower than Adam and AdamW, but much faster than standard SGD.

Standard SGD was the slowest and was still improving at Epoch `20`.

---

### Validation Performance

| Rank | Optimizer | Best Validation Loss | Final Validation Accuracy |
|---:|---|---:|---:|
| 1 | SGD with Momentum | 0.0814 | 97.44% |
| 2 | AdamW | 0.0985 | 97.36% |
| 3 | Adam | 0.0986 | 97.14% |
| 4 | SGD | 0.1101 | 96.94% |

SGD with Momentum achieved the best validation performance overall.

Adam and AdamW converged faster, but they did not achieve the best generalization.

---

### Stability

| Optimizer | Final Five-Epoch Val-Loss STD |
|---|---:|
| SGD with Momentum | 0.000657 |
| AdamW | 0.003487 |
| SGD | 0.003842 |
| Adam | 0.009132 |

SGD with Momentum had the most stable validation loss near the end of training.

Adam had the largest late-stage variation, which matches its stronger overfitting behavior.

However, this stability metric should not be interpreted alone. For example, SGD's validation loss was still decreasing, so its standard deviation does not necessarily mean it had already converged.

---

## 7. Optimizer Behavior in the Loss Landscape

SGD follows the current batch gradient directly using one global learning rate. This makes it simple but slower, especially in flat regions or narrow valleys.

SGD with Momentum adds a moving memory of previous update directions. This helps it move faster in consistent directions and reduces side-to-side oscillation.

Adam uses adaptive updates for each parameter based on gradient history. This explains its fast early convergence, but it can also overfit quickly if training continues after validation loss stops improving.

AdamW uses Adam's adaptive updates with separate weight decay. In this experiment, this helped reduce late overfitting compared with Adam.

---

## 8. Key Takeaway

Faster convergence did not guarantee better generalization.

Adam and AdamW reached their best validation-loss region the fastest, but SGD with Momentum achieved the best overall validation performance.

Standard SGD was the slowest and may need more epochs or learning-rate scheduling.

In this experiment, SGD with Momentum provided the best balance between convergence speed, stability, and validation performance.
