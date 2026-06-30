# Task 05 — Dropout Ablation Study

## 1. Objective

The objective of this task is to study how different Dropout rates affect overfitting and generalization. Three independent models were trained using the same architecture and settings:

* No Dropout
* Dropout = `0.1`
* Dropout = `0.3`

The training and validation losses were compared to determine how Dropout influences the model’s learned representations.

## 2. Code Used

```python
# Define the Dropout configurations.
dropout_rates = [0.0, 0.1, 0.3]
dropout_histories = {}

for dropout_rate in dropout_rates:

    # Create a new model for each experiment.
    dropout_model = create_dropout_model(
        dropout_rate=dropout_rate,
        seed=42
    )

    # Train every model using the same settings.
    history = dropout_model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # Store the training history.
    dropout_histories[dropout_rate] = history

    # Plot and save training loss versus validation loss.
    plot_dropout_loss(
        history,
        dropout_rate
    )
```

## 3. Results

| Configuration | Final Train Loss | Final Val Loss | Train Accuracy | Val Accuracy | Best Val Loss | Best Epoch |
| ------------- | ---------------: | -------------: | -------------: | -----------: | ------------: | ---------: |
| No Dropout    |           0.0089 |         0.1296 |         0.9973 |       0.9736 |        0.0979 |          5 |
| Dropout = 0.1 |           0.0344 |         0.1007 |         0.9879 |       0.9750 |        0.0858 |          8 |
| Dropout = 0.3 |           0.0888 |         0.0944 |         0.9709 |       0.9746 |        0.0853 |         12 |

### No Dropout

![No Dropout Loss Curve](../results/loss_curves/task05_dropout/task05_dropout_0_0_loss.png)

### Dropout = 0.1

![Dropout 0.1 Loss Curve](../results/loss_curves/task05_dropout/task05_dropout_0_1_loss.png)

### Dropout = 0.3

![Dropout 0.3 Loss Curve](../results/loss_curves/task05_dropout/task05_dropout_0_3_loss.png)

## 4. Short Analysis

### No Dropout — clear overfitting:

The loss gap is by far the widest at 0.1207, and the best validation loss occurred very early, at epoch 5. After that point, training loss kept falling (down to 0.0089) while validation loss climbed back up to 0.1296 — nearly 15× the final training loss. With no regularization, every neuron is free to fit the training set as tightly as possible, including its noise, and the model has no mechanism preventing it from drifting away from generalizable patterns once it passes the optimal point.

### Dropout = 0.1 — partial regularization:

The gap shrinks substantially to 0.0663, and the best epoch shifts later, to epoch 8 — dropout slowed the rate at which the model overfits, buying more useful training time. Final validation loss (0.1007) is noticeably lower than the no-dropout run, even though final training loss is higher (0.0344 vs 0.0089). This is the regularization tradeoff in action: a worse fit to training data in exchange for a better fit to unseen data.

### Dropout = 0.3 — best generalization, near-zero gap:

This configuration achieves the smallest loss gap of all three at just 0.0056, meaning training and validation loss converge to nearly the same value (0.0888 vs 0.0944). This is a strong sign of a well-regularized model: the network is no longer significantly outperforming itself on data it has memorized versus data it hasn't seen. The best epoch is also the latest of the three (epoch 12), showing dropout extended useful training time the most by continuously preventing the network from prematurely converging on memorized patterns. Despite having the highest final training loss of the three runs, this configuration achieved the best validation performance overall — both in terms of the lowest best validation loss (0.0853) and the tightest train-validation alignment.

### Comparison of Overfitting

| Configuration | Observed Overfitting |
| ------------- | -------------------- |
| No Dropout    | High                 |
| Dropout = 0.1 | Reduced              |
| Dropout = 0.3 | Lowest               |

The model without Dropout fitted the training data extremely closely, but its validation performance became worse. Both Dropout configurations reduced this gap, with `0.3` providing the strongest regularization.

However, training and validation losses under Dropout are not perfectly comparable because Dropout is active only while calculating the training metrics. Validation uses the complete network without randomly disabled neurons.

### How Dropout Prevents Neuron Co-adaptation

During training, Dropout randomly zeroes out a fraction of neuron outputs in a layer on each forward pass, forcing the remaining active neurons to produce useful outputs without relying on any specific combination of other neurons being present.

This breaks the tendency of neurons to co-adapt, where groups of neurons learn to compensate for each other’s specific errors rather than learning independently useful features.

Because the active subnetwork changes randomly during training, the network can be viewed approximately as an ensemble of many overlapping subnetworks that share the same weights. Each subnetwork is encouraged to learn robust and redundant representations.

The progression observed across the three experiments—a shrinking loss gap, a later best epoch, and close training–validation loss alignment with `Dropout = 0.3`—is consistent with the regularization effect becoming stronger as the Dropout rate increases.

## 5. Key Takeaway

Dropout reduced overfitting by preventing neurons from relying on fixed combinations of other neurons. A Dropout rate of `0.1` achieved the highest final validation accuracy, while `0.3` provided the strongest regularization and the lowest best validation loss.
