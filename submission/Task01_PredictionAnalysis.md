# Task 01 — Deep Prediction Analysis

## 1. Objective

The objective of this task is to analyze the predictions of a trained neural network on three samples from the test dataset. The analysis explains how the forward pass, activation functions, and the Adam optimizer contributed to each prediction.

## 2. Code Used

```python

def analyze_sample(sample_index):
    """
    Predict and display one sample from the test set.

    Parameters:
        sample_index (int): The index of the sample in x_test and y_test.
    """

    # Select the sample and its true label
    sample_x = x_test[sample_index]
    sample_y = y_test[sample_index]

    print("Sample index:", sample_index)
    print("Actual value:", sample_y)

    # Add a batch dimension: (28, 28) -> (1, 28, 28)
    sample_batch = sample_x.reshape(1, 28, 28)

    # Make the prediction
    prediction = model.predict(sample_batch, verbose=0)

    # Select the class with the highest probability
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)

    print("Predicted value:", predicted_label)
    print("Confidence:", confidence)

    # Check whether the prediction is correct
    if predicted_label == sample_y:
        print("Result: Correct prediction")
    else:
        print("Result: Incorrect prediction")

    # Display the image
    plt.imshow(sample_x, cmap="gray")

    plt.axis("off")
    plt.show()

# Call the function by passing the sample index:

# Sample 1
analyze_sample(666)

# Sample 2
analyze_sample(777)

# Sample 3
analyze_sample(888)

```

## 3. Results
The model produced a predicted label for each of the three selected test samples. The predicted label was obtained by selecting the output class with the highest probability.

### Sample 1

* **Predicted Label:** `7`
* **True Label:** `7`
* **Confidence:** `0.99839216`
* **Prediction Status:** `Correct`

### Sample 2

* **Predicted Label:** `1`
* **True Label:** `1`
* **Confidence:** `0.99873286`
* **Prediction Status:** `Correct`

### Sample 3

* **Predicted Label:** `2`
* **True Label:** `2`
* **Confidence:** `0.99996984`
* **Prediction Status:** `Correct`

## 4. Short Analysis

### How the forward pass works:
Each input sample passes through the network layer by layer. In a typical Dense network, every neuron computes a weighted sum of its inputs plus a bias: `z = W·x + b`. This raw value is then passed to an activation function.

```text
28 × 28 image
      ↓
Flatten layer
      ↓
784 pixel values
      ↓
Dense layer with 64 neurons
      ↓
ReLU activation
      ↓
Output layer with 10 neurons
      ↓
Softmax activation
      ↓
Predicted label
```

### Role of ReLU:
Hidden layers use ReLU (`f(z) = max(0, z)`), which introduces non-linearity while keeping gradients clean for positive values. ReLU allows the network to learn complex patterns without the vanishing gradient problem common with sigmoid or tanh.

### Role of Softmax:
The final layer uses Softmax, which converts raw logits into a probability distribution across all classes. The class with the highest probability becomes the predicted label. This is why we use `np.argmax()` on the output — we're selecting the most confident class.

### Adam optimizer's influence:
Adam adapts the learning rate for each weight individually using estimates of first and second moments of the gradients. During training, Adam helped the model converge faster and more stably than standard SGD. The high confidence scores (>0.97) indicate that the weights were well-optimized — Adam guided the model toward a sharp, confident decision boundary for these samples.

### Why predictions were correct:
The model likely saw many similar patterns during training. The features that activate strongly in earlier layers (edges, strokes, curves) were reliably mapped to the correct class through the learned weights. The high confidence scores suggest that Adam's adaptive updates pushed the weights toward a sharp, well-defined decision boundary for these sample classes.

## 5. Key Takeaway
The combination of ReLU activations for deep feature extraction and Softmax for probabilistic output, trained with Adam's adaptive updates, enabled the model to make confident and accurate predictions — showing that a well-trained network learns robust internal representations, not just memorized patterns.
