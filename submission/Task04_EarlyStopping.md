# Task 04 — EarlyStopping Behavior Analysis

## 1. Objective

The objective of this task is to examine how the `EarlyStopping` callback controls the training duration based on validation performance. The experiment compares `patience=3` and `patience=5`, and also investigates whether changing the optimizer from Adam to SGD affects the stopping pattern.

## 2. Code Used

```python
    # Experiment 1: Adam optimizer with patience = 3
    early_stopping_3 = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    model_patience_3 = create_model(optimizer="adam")
    
    history_patience_3 = model_patience_3.fit(
        x_train,
        y_train,
        epochs=30,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping_3],
        verbose=1
    )
    
    # Find the epoch where training stopped.
    stopped_epoch_3 = len(history_patience_3.history["loss"])
    
    # Find the epoch that achieved the lowest validation loss.
    best_epoch_3 = (
        np.argmin(history_patience_3.history["val_loss"]) + 1
    )
    
    best_val_loss_3 = np.min(
        history_patience_3.history["val_loss"]
    )
    
    print("Stopped epoch:", stopped_epoch_3)
    print("Best epoch:", best_epoch_3)
    print("Best validation loss:", best_val_loss_3)
    
    
    # Experiment 2: Adam optimizer with patience = 5
    early_stopping_5 = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_patience_5 = create_model(optimizer="adam")
    
    history_patience_5 = model_patience_5.fit(
        x_train,
        y_train,
        epochs=30,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping_5],
        verbose=1
    )
    
    stopped_epoch_5 = len(history_patience_5.history["loss"])
    
    best_epoch_5 = (
        np.argmin(history_patience_5.history["val_loss"]) + 1
    )
    
    best_val_loss_5 = np.min(
        history_patience_5.history["val_loss"]
    )
    
    print("Stopped epoch:", stopped_epoch_5)
    print("Best epoch:", best_epoch_5)
    print("Best validation loss:", best_val_loss_5)
    
    
    # Experiment 3: SGD optimizer with patience = 3
    sgd_optimizer = keras.optimizers.SGD(
        learning_rate=0.01
    )
    
    early_stopping_sgd = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    model_sgd = create_model(
        optimizer=sgd_optimizer
    )
    
    history_sgd = model_sgd.fit(
        x_train,
        y_train,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping_sgd],
        verbose=1
    )
    
    stopped_epoch_sgd = len(history_sgd.history["loss"])
    
    best_epoch_sgd = (
        np.argmin(history_sgd.history["val_loss"]) + 1
    )
    
    best_val_loss_sgd = np.min(
        history_sgd.history["val_loss"]
    )
    
    print("SGD stopped at epoch:", stopped_epoch_sgd)
    print("SGD best epoch:", best_epoch_sgd)
    print("Best validation loss:", best_val_loss_sgd)
```

## 3. Results

| Experiment | Optimizer | Patience | Maximum Epochs | Best Epoch | Training Ended At | Best Validation Loss | EarlyStopping Triggered |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | Adam | 3 | 30 | 4 | 7 | 0.0986 | Yes |
| 2 | Adam | 5 | 30 | 4 | 9 | 0.0986 | Yes |
| 3 | SGD | 3 | 50 | 50 | 50 | 0.0836 | No |

`Training Ended At` represents the number of completed epochs.

For the SGD experiment, training ended because it reached the maximum limit of `50 epochs`, not because `EarlyStopping` was triggered.

---

## 4. How the Best Epoch Was Determined

The best epoch was selected based on the minimum validation loss:

```python
best_epoch = np.argmin(history.history["val_loss"]) + 1
best_validation_loss = np.min(history.history["val_loss"])
```

`np.argmin()` returns the index of the lowest `val_loss`.

The value `+1` is added because Python indexing starts at `0`, while epoch numbering starts at `1`.

Validation loss was used because it measures performance on unseen validation data and can reveal overfitting when training loss continues decreasing.

---

## 5. Learning Curves

### Experiment 1 — Adam with Patience = 3

<table>
  <tr>
    <th>Loss Curves</th>
    <th>Accuracy Curves</th>
  </tr>
  <tr>
    <td>
      <img src="../results/loss_curves/task04_early_stopping/task04_adam_patience_3_loss.png" width="450">
    </td>
    <td>
      <img src="../results/loss_curves/task04_early_stopping/task04_adam_patience_3_accuracy.png" width="450">
    </td>
  </tr>
</table>

### Experiment 2 — Adam with Patience = 5

<table>
  <tr>
    <th>Loss Curves</th>
    <th>Accuracy Curves</th>
  </tr>
  <tr>
    <td>
      <img src="../results/loss_curves/task04_early_stopping/task04_adam_patience_5_loss.png" width="450">
    </td>
    <td>
      <img src="../results/loss_curves/task04_early_stopping/task04_adam_patience_5_accuracy.png" width="450">
    </td>
  </tr>
</table>

### Experiment 3 — SGD with Patience = 3

<table>
  <tr>
    <th>Loss Curves</th>
    <th>Accuracy Curves</th>
  </tr>
  <tr>
    <td>
      <img src="../results/loss_curves/task04_early_stopping/task04_sgd_patience_3_loss.png" width="450">
    </td>
    <td>
      <img src="../results/loss_curves/task04_early_stopping/task04_sgd_patience_3_accuracy.png" width="450">
    </td>
  </tr>
</table>

---

## 6. EarlyStopping Behavior Analysis

### Adam with Patience = 3

The lowest validation loss was `0.0986` at Epoch `4`.

After Epoch 4, validation loss did not improve for three consecutive epochs:

```text
Epochs without improvement: 5, 6, and 7
```

Therefore, `EarlyStopping` stopped training at Epoch `7`.

Because `restore_best_weights=True`, the final model restored the weights from Epoch 4 rather than keeping the weights from Epoch 7.

---

### Adam with Patience = 5

The lowest validation loss was also `0.0986` at Epoch `4`.

With `patience=5`, training continued for five consecutive epochs without improvement:

```text
Epochs without improvement: 5, 6, 7, 8, and 9
```

Training therefore stopped at Epoch `9`.

Increasing patience from `3` to `5` extended training by two epochs, but it did not produce a better validation loss. Both Adam experiments had the same best epoch and best validation loss.

The additional epochs reduced training loss further, while validation loss increased, showing that the model fitted the training data more strongly without improving generalization.

---

### Comparison of Patience Values

Because both Adam experiments used the same model initialization and training conditions, the effect of patience can be compared fairly.

| Patience | Best Epoch | Stopped Epoch | Best Validation Loss |
|---:|---:|---:|---:|
| 3 | 4 | 7 | 0.0986 |
| 5 | 4 | 9 | 0.0986 |

The larger patience value changed the stopping time but did not change the best model in this execution.

A larger patience value may help when validation loss temporarily fluctuates and later improves. However, it may also increase training time and allow overfitting to continue.

---

### Effect of Using SGD

SGD behaved differently from Adam.

Its validation loss decreased gradually throughout the experiment and reached its lowest value of `0.0836` at Epoch `50`.

Because the best result occurred at the final allowed epoch, there were no three consecutive epochs without improvement after it. Therefore:

```text
EarlyStopping Triggered: False
```

The actual EarlyStopping epoch for SGD cannot be determined from this experiment because training ended at the maximum limit of `50 epochs`.

SGD converged more slowly than Adam, but it continued improving and eventually achieved a lower validation loss in this execution:

```text
Adam Best Validation Loss: 0.0986
SGD Best Validation Loss:  0.0836
```

Adam reached its best result faster, while SGD improved more gradually over a larger number of epochs.

---

## 7. How the Optimizer Affected EarlyStopping

EarlyStopping does not directly depend on the optimizer. It only monitors the selected metric, which was `val_loss`.

However, the optimizer indirectly affects when EarlyStopping activates because it determines how the model weights are updated.

### Adam

Adam produced rapid improvement during the first few epochs.

The validation loss reached its minimum at Epoch 4 and then stopped improving. This caused EarlyStopping to activate early.

### SGD

SGD produced slower and more gradual changes.

The validation loss continued decreasing until the final epoch, so the required sequence of non-improving epochs never occurred.

This demonstrates that different optimizers can produce different stopping patterns even when EarlyStopping uses the same monitor and patience value.

---

## 8. EarlyStopping as Indirect Regularization

EarlyStopping acts as an indirect form of regularization by limiting how long the model trains.

When training continues after validation performance stops improving, the model may begin learning patterns specific to the training data.

EarlyStopping helps reduce this behavior by:

- Monitoring validation performance.
- Stopping after a selected number of non-improving epochs.
- Restoring the weights from the best validation epoch.

Unlike L2 regularization, it does not add a penalty to the loss function. Unlike Dropout, it does not change the network architecture.

---

## 9. Important Note About Restored Weights

The final metrics stored in `history.history` belong to the last executed epoch.

However, when:

```python
restore_best_weights=True
```

the model weights are restored after training to the best epoch.

Therefore, for the Adam experiments:

- The final recorded history values belong to Epoch 7 or Epoch 9.
- The final model weights belong to Epoch 4.

To measure the restored model directly, it can be evaluated after training using:

```python
model.evaluate(x_val, y_val)
```

---

## 10. Key Takeaway

EarlyStopping successfully stopped both Adam experiments after validation loss stopped improving.

Increasing patience from `3` to `5` delayed stopping from Epoch `7` to Epoch `9`, but both experiments restored the same best weights from Epoch `4`.

Adam reached its best validation result quickly, while SGD improved more gradually and did not trigger EarlyStopping within the `50-epoch` limit.

The experiment shows that patience controls how long the model waits, while the optimizer influences the validation-loss behavior that EarlyStopping monitors.
