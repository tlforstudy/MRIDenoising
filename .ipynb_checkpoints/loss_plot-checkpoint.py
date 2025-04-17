import json
import matplotlib.pyplot as plt


with open('./results/training_history.json', 'r') as f:
    history = json.load(f)


train_loss = history["loss"]
val_loss = history["val_loss"]
epochs = list(range(1, len(train_loss) + 1)) 



plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, "b-", label="Training Loss")
plt.plot(epochs, val_loss, "r-", label="Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (DSSIM)")
plt.legend()
plt.savefig("./results/loss_curve.png", dpi=300)
plt.show()