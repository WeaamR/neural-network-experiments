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

`Training Ended At` represents the number of epochs completed.

For the SGD experiment, training ended because it reached the maximum limit of `50 epochs`, not because `EarlyStopping` was triggered.

---

## 4. How the Best Epoch Was Determined

The best epoch was selected based on the minimum validation loss:

```python
best_epoch = np.argmin(history.history["val_loss"]) + 1
best_validation_loss = np.min(history.history["val_loss"])
