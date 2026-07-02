# Task 08 — Batch Size & Gradient Noise Experiment

## 1. Objective

The objective of this experiment is to study how changing the batch size affects:

- Gradient noise.
- Training speed.
- Generalization.
- Overfitting.
- The smoothness of the loss curves.

Three independent models were trained using:

- Batch Size = `8`
- Batch Size = `32`
- Batch Size = `128`

All other settings, including the model architecture, optimizer, learning rate, dataset, random seed, and number of epochs, were kept constant.

---

## 2.  Code Used

```python
# Create a directory for Task 8 results.

def create_batch_size_model(seed=42):

    # Ensure reproducible initial weights.
    keras.utils.set_random_seed(seed)

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),

        # Convert each 28×28 image into 784 values.
        keras.layers.Flatten(),

        # Hidden layer.
        keras.layers.Dense(64, activation="relu"),

        # Output layer for digits 0–9.
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_curves(history, label, save_dir, file_prefix, metric):
    # Plot and save training vs validation curves for a given metric.
    epoch_range = range(1, len(history.history[metric]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epoch_range, history.history[metric],          label=f"Training {metric.capitalize()}")
    plt.plot(epoch_range, history.history[f"val_{metric}"], label=f"Validation {metric.capitalize()}")
    plt.title(f"{metric.capitalize()} Curves — {label}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid()

    output_path = save_dir / f"{file_prefix}_{metric}.png"

    # Save the figure before displaying it.
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {output_path}")


# Train all three batch-size configurations.
batch_sizes      = [8, 32, 128]
number_of_epochs = 20
batch_histories  = {}
batch_results    = {}

for batch_size in batch_sizes:

    label       = f"Batch Size = {batch_size}"
    file_prefix = f"batch_{batch_size}"

    # Create a fresh model for this experiment.
    model = create_batch_size_model(seed=42)

    # Calculate the number of weight updates per epoch.
    steps_per_epoch = math.ceil(len(x_train) / batch_size)

    # Measure the total training time.
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=number_of_epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        shuffle=True,
        verbose=1
    )

    training_time = time.time() - start_time

    # Store the history for the comparison plot.
    batch_histories[batch_size] = history

    # Create and save the loss and accuracy plots.
    plot_curves(history, label, task8_results_dir, file_prefix, "loss")
    plot_curves(history, label, task8_results_dir, file_prefix, "accuracy")

    # Find the lowest validation loss and its epoch.
    best_val_loss       = np.min(history.history["val_loss"])
    best_val_loss_epoch = np.argmin(history.history["val_loss"]) + 1

    # Measure validation-loss variation over the final five epochs.
    final_five_val_loss_std = np.std(history.history["val_loss"][-5:])

    # Store the main results needed for comparison.
    batch_results[batch_size] = {
        "final_train_loss":       history.history["loss"][-1],
        "final_val_loss":         history.history["val_loss"][-1],
        "final_val_accuracy":     history.history["val_accuracy"][-1],
        "best_val_loss":          best_val_loss,
        "best_val_loss_epoch":    best_val_loss_epoch,
        "steps_per_epoch":        steps_per_epoch,
        "final_five_val_loss_std":final_five_val_loss_std,
        "training_time":          training_time
    }


# Plot validation-loss comparison across all batch sizes.
plt.figure(figsize=(8, 5))
for batch_size, history in batch_histories.items():
    epoch_range = range(1, len(history.history["val_loss"]) + 1)
    plt.plot(epoch_range, history.history["val_loss"], label=f"Batch Size = {batch_size}")

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid()

comparison_path = task8_results_dir / "batch_size_validation_loss_comparison.png"
plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {comparison_path}")


# Print and save all results.
results_file = task8_results_dir / "task08_batch_size_results.txt"

with open(results_file, "w", encoding="utf-8") as f:
    f.write("Task 08 — Batch Size & Gradient Noise Experiment\n")
    f.write("=" * 55 + "\n")

    for batch_size, r in batch_results.items():
        line = (
            f"\nBatch Size = {batch_size}\n"
            f"Final Training Loss:               {r['final_train_loss']:.4f}\n"
            f"Final Validation Loss:             {r['final_val_loss']:.4f}\n"
            f"Final Validation Accuracy:         {r['final_val_accuracy']:.4f}\n"
            f"Best Validation Loss:              {r['best_val_loss']:.4f}\n"
            f"Best Validation Loss Epoch:        {r['best_val_loss_epoch']}\n"
            f"Steps Per Epoch:                   {r['steps_per_epoch']}\n"
            f"Final 5-Epoch Validation Loss STD: {r['final_five_val_loss_std']:.6f}\n"
            f"Training Time:                     {r['training_time']:.2f} seconds\n"
        )
        print(line)
        f.write(line)

print(f"Results saved to: {results_file}")
```
---

## 3. Results

| Batch Size | Final Train Loss | Final Val Loss | Final Val Accuracy | Best Val Loss | Best Epoch | Steps per Epoch | Final 5-Epoch Val-Loss STD | Training Time |
|-----------:|-----------------:|---------------:|-------------------:|--------------:|-----------:|----------------:|---------------------------:|--------------:|
| `8`   | 0.0120 | 0.1608 | 97.60% | 0.0942 | 4  | 6875 | 0.018815 | 362.47 s |
| `32`  | 0.0083 | 0.1304 | 97.52% | 0.0986 | 4  | 1719 | 0.001615 | 99.35 s |
| `128` | 0.0228 | 0.1050 | 96.90% | 0.0905 | 10 | 430  | 0.003218 | 32.76 s |

### Validation Loss Comparison

![Validation Loss Comparison](../results/batch_size_tests/task08_batch_sizes/batch_size_validation_loss_comparison.png)

---

## 3. Loss and Accuracy Curves

### Batch Size = 8

![Batch Size 8 Loss Curves](../results/batch_size_tests/task08_batch_sizes/batch_8_loss.png)

![Batch Size 8 Accuracy Curves](../results/batch_size_tests/task08_batch_sizes/batch_8_accuracy.png)

### Batch Size = 32

![Batch Size 32 Loss Curves](../results/batch_size_tests/task08_batch_sizes/batch_32_loss.png)

![Batch Size 32 Accuracy Curves](../results/batch_size_tests/task08_batch_sizes/batch_32_accuracy.png)

### Batch Size = 128

![Batch Size 128 Loss Curves](../results/batch_size_tests/task08_batch_sizes/batch_128_loss.png)

![Batch Size 128 Accuracy Curves](../results/batch_size_tests/task08_batch_sizes/batch_128_accuracy.png)

---

## 4. Short Analysis

### Batch Size = 8 — Many Updates and Early Overfitting

Batch Size `8` performed `6875` parameter updates per epoch, which was the largest number among the three configurations.

It reached its best validation loss of `0.0942` at epoch `4`. After that point, the training loss continued decreasing until it reached `0.0120`, while the validation loss increased substantially to `0.1608`.

The validation-loss curve also showed a large fluctuation near epoch `18`. Its final five-epoch validation-loss standard deviation was `0.018815`, the highest among the three configurations.

This indicates that the small batch produced less stable validation behavior and began overfitting early.

However, it achieved the highest final validation accuracy of `97.60%`. The combination of increasing validation loss and stable accuracy suggests that the model may have become more confident in some incorrect predictions, causing Cross-Entropy Loss to increase even when the number of correct predictions remained high.

It also required the longest training time, `362.47` seconds, because the small batch caused a very large number of weight updates.

---

### Batch Size = 32 — Balanced and Stable Behavior

Batch Size `32` performed `1719` updates per epoch.

It reached its best validation loss of `0.0986` at epoch `4`. Its training loss continued decreasing after that point, while validation loss gradually increased to `0.1304`, indicating overfitting.

However, its validation-loss curve was considerably more stable than Batch Size `8`.

The final five-epoch validation-loss standard deviation was only `0.001615`, the lowest among the tested configurations.

This configuration provided a practical balance between:

- Training speed.
- Gradient stability.
- Validation accuracy.
- Computational cost.

Its total training time was `99.35` seconds, substantially lower than Batch Size `8`.

---

### Batch Size = 128 — Fastest Training and Lowest Best Validation Loss

Batch Size `128` required only `430` updates per epoch.

It completed training in `32.76` seconds, making it the fastest configuration by a large margin.

This model reached the lowest best validation loss among all three experiments:

```text
Best Validation Loss = 0.0905
Best Epoch = 10
```

Its validation loss improved for more epochs than the smaller batches before beginning to increase.

It also produced the smallest final train–validation loss gap:

```text
0.1050 - 0.0228 = 0.0822
```

However, its final validation accuracy was `96.90%`, which was slightly lower than the validation accuracies achieved with Batch Sizes `8` and `32`.

Therefore, the larger batch achieved better validation loss and faster computation, but slightly lower classification accuracy.

---

## 5. Why Smaller Batches Introduce Gradient Noise

The gradient used in each weight update is estimated from the samples in the current batch.

With Batch Size `8`, the estimate is based on only eight training examples. These examples may not represent the complete training-data distribution.

As a result, the gradient direction can vary considerably between batches:

```text
Batch 1 → Gradient direction A
Batch 2 → Gradient direction B
Batch 3 → Gradient direction C
```

This variation is called **gradient noise**.

A larger batch averages the gradients of more samples, producing an estimate that is generally closer to the gradient of the complete dataset:

```text
Small batch → Noisier gradient estimate
Large batch → More stable gradient estimate
```

---

## 6. When Gradient Noise Can Be Beneficial

Gradient noise is not always harmful.

The random variation in small-batch updates may help the optimizer move away from:

- Saddle points.
- Narrow regions of the loss landscape.
- Sharp minima.
- Poor optimization paths.

Instead of following one highly stable direction, the optimizer explores nearby regions of the loss landscape.

This exploration may help the model reach flatter solutions that are less sensitive to small changes in the input and may generalize better.

However, excessive noise can also make training unstable. Therefore, a small batch does not automatically guarantee better generalization.

In this experiment, Batch Size `8` achieved the highest final validation accuracy, but it also showed the strongest late-stage increase and fluctuation in validation loss.

---

## 7. Why Larger Batches Can Train Faster

A larger batch processes more samples in each update and requires fewer updates to complete one epoch.

The experiments produced:

```text
Batch Size = 8   → 6875 steps per epoch
Batch Size = 32  → 1719 steps per epoch
Batch Size = 128 → 430 steps per epoch
```

Therefore, Batch Size `128` completed far fewer optimizer steps than Batch Size `8`.

Larger batches can also use GPU parallel processing more efficiently, which reduces training time.

The measured training times clearly demonstrated this:

```text
Batch Size = 8   → 362.47 seconds
Batch Size = 32  → 99.35 seconds
Batch Size = 128 → 32.76 seconds
```

However, faster wall-clock training does not necessarily mean that the model reaches its best validation result in fewer epochs.

Batch Size `128` was fastest in time, but its best validation loss occurred at epoch `10`, compared with epoch `4` for the smaller batches.

---

## 8. Why Larger Batches May Generalize Worse

A large batch produces a more accurate and stable gradient estimate.

This stability allows the optimizer to move directly toward a minimum, but it may reduce exploration of alternative regions in the loss landscape.

In some cases, this can lead the model toward a sharp minimum that performs well on training data but is more sensitive to unseen data.

Small-batch noise can act as a form of implicit regularization by preventing the model from following an overly precise optimization path.

However, this behavior is not guaranteed.

In this experiment, Batch Size `128` achieved the best validation loss and the smallest final loss gap, but its validation accuracy was slightly lower than the smaller-batch configurations.

Therefore, the claim that larger batches always generalize worse was not fully supported by these results. The conclusion depended on whether generalization was measured using validation loss or validation accuracy.

---

## 9. Effect of Batch Size on Loss-Curve Smoothness

The validation-loss curves showed different levels of smoothness.

### Batch Size = 8

The curve was the most irregular during later epochs and contained a large spike near epoch `18`.

Its final five-epoch validation-loss standard deviation was:

```text
0.018815
```

### Batch Size = 32

The validation-loss curve changed more gradually and had the lowest late-stage variability:

```text
0.001615
```

### Batch Size = 128

The curve was also relatively smooth, although its late-stage variability was slightly greater than Batch Size `32`:

```text
0.003218
```

Therefore, the results showed that Batch Size `32` produced the smoothest validation-loss trend, while Batch Size `8` produced the least stable trend.

Loss-curve smoothness is influenced not only by batch size, but also by the optimizer, learning rate, data order, and the stage of training.

---

## 10. Important Experimental Consideration

Using the same number of epochs does not give every batch size the same number of parameter updates.

Over 20 epochs:

```text
Batch Size = 8   → 137500 updates
Batch Size = 32  → 34380 updates
Batch Size = 128 → 8600 updates
```

Therefore, Batch Size `8` received approximately sixteen times more updates than Batch Size `128`.

This large difference helps explain why the smaller batch fitted the training set more aggressively and began overfitting earlier.

A stricter comparison could control the total number of optimizer updates rather than only controlling the number of epochs.

---

## 11. Key Takeaway

Batch Size `8` produced the largest number of weight updates, the longest training time, and the most irregular validation-loss behavior. It achieved the highest final validation accuracy but also showed strong late-stage overfitting.

Batch Size `32` provided the most balanced behavior and the smoothest late-stage validation-loss curve.

Batch Size `128` was the fastest configuration and achieved the lowest best validation loss, but its validation accuracy was slightly lower.

The experiment demonstrates that batch size creates a tradeoff between gradient noise, computational speed, optimization stability, and generalization. No single batch size is always best; the most suitable choice depends on the model, optimizer, dataset, and evaluation metric.
