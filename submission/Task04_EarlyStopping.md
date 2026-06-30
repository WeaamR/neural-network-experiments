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

| Experiment | Optimizer | Patience | Best Epoch | Stopped Epoch | Best Validation Loss |
| ---------- | --------- | -------: | ---------: | ------------: | -------------------: |
| 1          | Adam      |        3 |  `7` |     `10` |            `0.0822` |
| 2          | Adam      |        5 |  `8` |     `13` |            `0.0775` |
| 3          | SGD       |        3 |  `49` |     `50` |            `0.0859` |

## 4. Short Analysis

### Adam with Patience = 3

The model achieved its lowest validation loss of `0.0822` at epoch `7`. Training then continued for three consecutive epochs without further improvement and stopped at epoch `10`.

This matches the behavior of `patience=3`: EarlyStopping waits for three epochs after the best result before stopping. Because `restore_best_weights=True` was enabled, the model restored the weights learned at epoch `7`, rather than keeping the weights from epoch `10`.

### Why Validation Loss Controls the Decision

Validation loss measures the model’s performance on data that is not used to update its weights. Training loss may continue decreasing even when the model starts fitting the training data too closely.

Therefore, monitoring `val_loss` helps detect when additional training is no longer improving generalization. When validation loss stops decreasing, the model may be approaching or beginning to overfit.

### Adam with Patience = 5

With `patience=5`, the lowest validation loss was `0.0775` at epoch `8`, and training stopped at epoch `13`.

The larger patience value allowed the model to continue for five epochs without improvement. Compared with `patience=3`, this gave the model more time to recover from temporary fluctuations in validation loss.

In this experiment, the longer patience produced a slightly lower best validation loss. However, increasing patience does not always guarantee better performance; it may also increase training time and allow overfitting to continue for longer.

### Effect of Using SGD

With SGD, the best validation loss was `0.0859` at epoch `49`. Training ended at epoch `50`, but this was not caused by EarlyStopping.

Since only one epoch occurred after the best epoch, the required three consecutive epochs without improvement were not reached. The experiment ended because it reached the maximum limit of `50 epochs`.

This shows that **SGD converged more slowly than Adam** in this experiment. Adam reached its best validation performance within the first eight epochs, while SGD continued improving gradually until epoch `49`.

A different optimizer changes the EarlyStopping pattern indirectly because it affects the speed and stability of the weight updates. EarlyStopping still monitors the same `val_loss`, but different optimizers produce different validation-loss curves.

### EarlyStopping as Indirect Regularization

EarlyStopping acts as an indirect form of regularization by limiting the number of weight updates performed during training.

When training continues for too long, the model may begin learning noise or details that are specific to the training set. Stopping near the best validation epoch helps prevent this excessive fitting and preserves better generalization.

Unlike L2 regularization, EarlyStopping does not add a penalty to the loss function. Unlike Dropout, it does not change the network architecture. Instead, it controls how long the model is allowed to train.

## 5. Key Takeaway

Adam converged much faster than SGD, while increasing patience from `3` to `5` allowed more time for validation performance to improve. EarlyStopping helps reduce overfitting by stopping training when validation loss no longer improves and restoring the best learned weights.
