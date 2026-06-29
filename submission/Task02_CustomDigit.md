## Task 02 — Custom Image Generalization Test

### 1. Objective

Draw a custom handwritten digit, preprocess it to match the model's expected input format, run a prediction, and analyze whether the model generalizes beyond the training distribution — referencing representation learning, distribution shift, and augmentation.

---

### 2. Code Used

```python
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Load the custom drawn image
img = Image.open("custom_digit.png").convert("L")   # convert to grayscale

# Resize to 28x28 (MNIST format)
img = img.resize((28, 28))

# MNIST uses white digit on black background
# If your drawing is black on white, invert it
img = ImageOps.invert(img)

# Normalize to [0, 1]
img_array = np.array(img) / 255.0

# Reshape to match model input: (1, 28, 28) or (1, 784) depending on your model
img_input = img_array.reshape(1, -1)   # for Flatten input
# img_input = img_array.reshape(1, 28, 28, 1)  # for Conv2D input

# Preview the processed image
plt.imshow(img_array, cmap="gray")
plt.title("Custom Input After Preprocessing")
plt.axis("off")
plt.savefig("results/task2_custom_input.png")
plt.show()

# Run prediction
prediction = model.predict(img_input)
predicted_label = np.argmax(prediction)
confidence = prediction[0][predicted_label]

print(f"Predicted Label : {predicted_label}")
print(f"Confidence      : {confidence:.4f}")
```

---

### 3. Results

```
Predicted Label : 3
Confidence      : 0.6712
```

*(Replace with your actual output)*

> **Plot:** Preprocessed custom digit saved to `results/task2_custom_input.png`

---

### 4. Short Analysis

**Did the model classify correctly?**
*(Update based on your result — use one of the two cases below)*

**✅ If correct:**
The model successfully classified the custom digit, which suggests the learned internal representations are robust enough to handle slight variations in stroke style and thickness. The digit's key structural features — such as curves and line angles — were close enough to the training distribution for the model to activate the correct class neurons confidently.

**❌ If incorrect:**
The model failed to classify the custom digit correctly, and this is expected behavior that reveals an important limitation.

**Distribution shift:**
The MNIST dataset contains digits written under specific conditions — consistent stroke thickness, centered positioning, and clean backgrounds. A digit drawn with a mouse or stylus may differ in stroke texture, line weight, or positioning, causing a mismatch between the training distribution and the custom input. The model was never exposed to this style, so its learned filters activate incorrectly.

**Representation learning:**
Neural networks do not learn the abstract concept of "what a digit means." Instead, they learn statistical patterns — pixel-level features like edges, curves, and junctions — that appeared consistently in training data. If the custom digit's pixel structure deviates from those patterns, the model's intermediate layers fail to activate the correct feature detectors, and the final Softmax output becomes uncertain or wrong.

**Lack of augmentation:**
If the model was trained without data augmentation (rotations, shifts, noise), it has no exposure to drawing-style variation. A model trained with augmentation would be significantly more robust to custom inputs.

---

### 5. Key Takeaway

A model that performs well on test data does not automatically generalize to real-world custom inputs — distribution shift between training data and custom drawings exposes the limits of representation learning when augmentation is absent.

---

ملاحظتان قبل الرفع:

أولاً، في قسم Results، شغّل الكود وحط النتيجة الفعلية، وإذا الـ prediction غلط احذف الـ ✅ section وخلي بس الـ ❌، والعكس.

ثانياً، تأكد إن الـ background صح — MNIST أبيض على أسود، فإذا رسمت بالعكس لازم `ImageOps.invert()` تشتغل، وإلا الموديل ما رح يتعرف عليه أصلاً.
