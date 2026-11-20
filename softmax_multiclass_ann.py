"""
ANN Practical Assignment - Roll No. 8041
Topic  : Multi-Class Classification with Softmax
Subject: Artificial Neural Networks (ANN)

A feed-forward neural network built from scratch using NumPy:
    Input -> Hidden layer (ReLU) -> Output layer (Softmax)
Loss      : Categorical cross-entropy
Optimizer : Mini-batch gradient descent
Dataset   : Synthetic 3-class data (Gaussian blobs), so no downloads are needed.

Run:  python softmax_multiclass_ann.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 8041
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- data
def make_data(n_per_class=200, n_classes=3):
    centers = np.array([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0]])[:n_classes]
    X, y = [], []
    for c, mu in enumerate(centers):
        X.append(rng.normal(mu, 1.0, size=(n_per_class, 2)))
        y.append(np.full(n_per_class, c))
    X, y = np.vstack(X), np.concatenate(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def one_hot(y, k):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


def train_test_split(X, y, test_size=0.2):
    n_test = int(len(X) * test_size)
    return X[n_test:], X[:n_test], y[n_test:], y[:n_test]


# ---------------------------------------------------- activations / loss
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)      # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(y_true_oh, probs):
    return -np.mean(np.sum(y_true_oh * np.log(probs + 1e-12), axis=1))


# ---------------------------------------------------------------- model
class SoftmaxANN:
    def __init__(self, n_in, n_hidden, n_out, lr=0.1):
        self.lr = lr
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def backward(self, y_oh):
        m = len(y_oh)
        dz2 = (self.probs - y_oh) / m            # softmax + cross-entropy gradient
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def fit(self, X, y, epochs=200, batch_size=32):
        k = self.W2.shape[1]
        y_oh = one_hot(y, k)
        history = []
        for ep in range(1, epochs + 1):
            idx = rng.permutation(len(X))
            for i in range(0, len(X), batch_size):
                b = idx[i:i + batch_size]
                self.forward(X[b])
                self.backward(y_oh[b])
            loss = cross_entropy(y_oh, self.forward(X))
            acc = np.mean(self.predict(X) == y)
            history.append((loss, acc))
            if ep % 20 == 0 or ep == 1:
                print(f"Epoch {ep:4d} | loss = {loss:.4f} | train acc = {acc:.4f}")
        return history


# ------------------------------------------------------------- metrics
def confusion_matrix(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_results(model, X, y, history, path_prefix="output"):
    losses, accs = zip(*history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(losses); ax[0].set_title("Training Loss"); ax[0].set_xlabel("Epoch")
    ax[1].plot(accs);   ax[1].set_title("Training Accuracy"); ax[1].set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig(f"{path_prefix}_training_curves.png", dpi=150); plt.close()

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=20)
    plt.title("Decision Boundaries (Softmax ANN)")
    plt.savefig(f"{path_prefix}_decision_boundary.png", dpi=150); plt.close()


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    K = 3
    X, y = make_data(n_per_class=200, n_classes=K)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mu, sd = X_train.mean(0), X_train.std(0)      # standardise using train stats
    X_train, X_test = (X_train - mu) / sd, (X_test - mu) / sd

    model = SoftmaxANN(n_in=2, n_hidden=16, n_out=K, lr=0.1)
    history = model.fit(X_train, y_train, epochs=200, batch_size=32)

    y_pred = model.predict(X_test)
    print("\nTest accuracy:", round(float(np.mean(y_pred == y_test)), 4))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, y_pred, K))

    plot_results(model, X_train, y_train, history)
    print("Saved: output_training_curves.png, output_decision_boundary.png")
