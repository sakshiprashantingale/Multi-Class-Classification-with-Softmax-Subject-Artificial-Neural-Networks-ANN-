# Multi-Class Classification with Softmax (ANN Practical)

**Roll No.:** 8041  
**Subject:** Artificial Neural Networks (ANN)

## Aim
To implement a neural network for multi-class classification using the softmax
activation function and categorical cross-entropy loss.

## Theory (short)
- **Softmax:** softmax(z_i) = e^(z_i) / Σ_j e^(z_j). It turns raw outputs into class probabilities that sum to 1.
- **Loss:** categorical cross-entropy, L = -Σ y_i log(p_i).
- **Gradient:** for softmax + cross-entropy, dL/dz = p - y.
- **Architecture:** Input (2) → Hidden (16, ReLU) → Output (3, Softmax).

## Requirements
```
pip install numpy matplotlib
```

## Run
```
python softmax_multiclass_ann.py
```

## Output
- Training loss/accuracy printed every 20 epochs
- Test accuracy and confusion matrix
- `output_training_curves.png`
- `output_decision_boundary.png`

## Conclusion
The softmax ANN separates the three classes and reaches high test accuracy.
