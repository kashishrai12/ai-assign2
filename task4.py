import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# -----------------------------
# Load MNIST Subset (5 Classes)
# -----------------------------
classes = [0, 1, 2, 3, 4]

transform = transforms.ToTensor()

mnist = datasets.MNIST(root="./data", train=True,
                      download=True, transform=transform)

X, Y = [], []
for img, label in mnist:
    if label in classes:
        X.append(img.view(-1).numpy())
        Y.append(classes.index(label))

X = np.array(X)
Y = np.array(Y)

# Train/Test Split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = Y[:split], Y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# FCNN Architecture (4 Hidden Layers)
# -----------------------------
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Train Function with Stopping
# -----------------------------
def train_model(optimizer_name):
    model = FCNN()
    criterion = nn.CrossEntropyLoss()

    lr = 0.001

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)

    elif optimizer_name == "BatchGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)

    elif optimizer_name == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif optimizer_name == "NAG":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, nesterov=True)

    elif optimizer_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=lr,
                                  alpha=0.99, eps=1e-8)

    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               betas=(0.9,0.999), eps=1e-8)

    prev_loss = float("inf")
    epochs = 0

    while True:
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        epochs += 1

        if abs(prev_loss - loss.item()) < 1e-4:
            break

        prev_loss = loss.item()

        if epochs % 10 == 0:
            print(f"{optimizer_name} Epoch {epochs}, Loss={loss.item():.4f}")

    # Test Accuracy
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)

    acc = accuracy_score(y_test.numpy(), preds.numpy())
    cm = confusion_matrix(y_test.numpy(), preds.numpy())

    return epochs, acc, cm

# -----------------------------
# Run All Optimizers
# -----------------------------
optimizers = ["SGD", "BatchGD", "Momentum", "NAG", "RMSProp", "Adam"]

results = {}

for opt in optimizers:
    print("\nTraining with:", opt)
    ep, acc, cm = train_model(opt)
    results[opt] = (ep, acc)

    print("Converged Epochs:", ep)
    print("Test Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

# Print Summary Table
print("\n===== Final Comparison =====")
for opt in results:
    print(opt, "Epochs:", results[opt][0],
          "Accuracy:", results[opt][1])