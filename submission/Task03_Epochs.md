## Task 03 — Epoch-Based Learning Curve Exploration

### 1. Objective

Train the same model three times using 5, 10, and 20 epochs, plot the learning curves for each run, and analyze the training dynamics — identifying overfitting signs and explaining how Adam influenced convergence speed and stability.

---

### 2. Code Used

```python
# Create a new neural network model for each experiment.
# Each model starts with newly initialized weights.
def create_model(optimizer="adam"):

    model = keras.Sequential([
        # Define the expected input shape of an MNIST image.
        keras.layers.Input(shape=(28, 28)),

        # Convert the 28×28 image into a vector of 784 pixel values.
        keras.layers.Flatten(),

        # Hidden layer that learns useful non-linear patterns.
        keras.layers.Dense(64, activation="relu"),

        # Output layer with one probability for each digit from 0 to 9.
        keras.layers.Dense(10, activation="softmax")
    ])

    # Configure the model for training.
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Create a directory for Task 3 results.
task3_results_dir = Path("results/loss_curves/task03_epochs")
task3_results_dir.mkdir(parents=True, exist_ok=True)

def plot_learning_curves(history, num_epochs):

    # Plot training loss and validation loss.
    plt.figure(figsize=(7, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(f"Loss Curves — {num_epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Save the loss curve before displaying it
    plt.savefig(
        task3_results_dir / f"task03_{num_epochs}_epochs_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Plot training accuracy and validation accuracy.
    plt.figure(figsize=(7, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(f"Accuracy Curves — {num_epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    # Save the accuracy curve before displaying it.
    plt.savefig(
        task3_results_dir / f"task03_{num_epochs}_epochs_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# Experiment 1: Train a new model for 5 epochs.
model_5 = create_model()

history_5 = model_5.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_val, y_val)

# Experiment 2: Train a new model for 10 epochs.
model_10 = create_model()

history_10 = model_10.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_val, y_val)
)

# Experiment 3: Train a new model for 20 epochs.
model_20 = create_model()

history_20 = model_20.fit(
    x_train,
    y_train,
    epochs=20,
    validation_data=(x_val, y_val)
)

# Store the training histories in a dictionary.
histories = {
    5: history_5,
    10: history_10,
    20: history_20
}

# Create the directory for Task 3 results if it does not exist.
task3_results_dir = Path("results/loss_curves/task03_epochs")
task3_results_dir.mkdir(parents=True, exist_ok=True)

# Define the path of the text file.
results_file = task3_results_dir / "task03_final_metrics.txt"

# Save the final metrics of each experiment.
with open(results_file, "w", encoding="utf-8") as file:

    file.write("Task 03 — Epoch-Based Learning Curve Exploration\n")
    file.write("=" * 55 + "\n")

    for epochs, history in histories.items():

        # Get the metrics recorded at the final epoch.
        final_training_loss = history.history["loss"][-1]
        final_validation_loss = history.history["val_loss"][-1]
        final_training_accuracy = history.history["accuracy"][-1]
        final_validation_accuracy = history.history["val_accuracy"][-1]

        file.write(f"\nResults for {epochs} Epochs\n")
        file.write("-" * 30 + "\n")
        file.write(f"Final Training Loss: {final_training_loss:.4f}\n")
        file.write(f"Final Validation Loss: {final_validation_loss:.4f}\n")
        file.write(f"Final Training Accuracy: {final_training_accuracy:.4f}\n")
        file.write(f"Final Validation Accuracy: {final_validation_accuracy:.4f}\n")

```

---

### 3. Results

| Epochs | Train Loss | Val Loss | Train Acc | Val Acc |
|--------|-----------|----------|-----------|---------|
| 5      | 0.0713    | 0.0958   | 0.9783    | 0.9732  |
| 10     | 0.0290    | 0.0887   | 0.9911    | 0.9772  |
| 20     | 0.0091    | 0.1131   | 0.9973    | 0.9782  |

> **Plots saved to** `results/` — loss and accuracy curves for each of the three runs.

---

### 4. Short Analysis

**5 Epochs — Underfitting zone:**

Both curves are still descending at epoch 5 with no gap between them, which means the model has not yet converged. Training accuracy of 0.9783 vs. validation accuracy of 0.9732 shows the two are still close — a sign that the model is still learning generalizable features and has not started memorizing. This is the healthy early-training phase, but stopping here leaves performance on the table.

**10 Epochs — Sweet spot:**

At 10 epochs, the training loss (0.0290) is notably lower than validation loss (0.0887), and a small gap begins to appear between the two curves. However, validation accuracy (0.9772) is still improving and the gap is not widening aggressively. This is the most balanced run — the model has learned robust representations without significantly overfitting.

**20 Epochs — Early overfitting signal:**

This is where the divergence becomes clear. Training loss dropped to 0.0091 while validation loss actually increased to 0.1131 — higher than at 10 epochs. Looking at the loss curve, the validation loss plateaus and starts drifting upward after approximately epoch 5–7, while training loss continues to fall. Training accuracy reached 0.9973 but validation accuracy only gained marginally (0.9782), a difference of nearly 2% — a classic overfitting signature. The model is memorizing training-specific patterns rather than learning generalizable features.

**Adam optimizer's role:**

Across all three runs, Adam drove fast and stable convergence in the early epochs — the sharp drop in training loss between epoch 0 and epoch 3 is visible in all three plots. This is Adam's adaptive learning rate at work: it automatically scales updates per-weight based on gradient history, allowing the model to make large steps early when gradients are strong and smaller, more precise steps as it approaches a minimum. This is why even the 5-epoch run already achieves 97%+ accuracy — Adam compresses the useful learning into very few epochs. However, Adam's efficiency also means the model reaches the overfitting threshold faster, which is why the validation loss starts rising noticeably by epoch 7–8 in the 20-epoch run without any regularization in place.

---

### 5. Key Takeaway

More epochs do not always mean better performance — beyond a certain point, training loss keeps falling while validation loss rises, revealing overfitting; the 10-epoch run represents the optimal balance between convergence and generalization for this model and dataset.
