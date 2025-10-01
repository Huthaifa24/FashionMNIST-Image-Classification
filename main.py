import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from tqdm.auto import tqdm #for cool progress bars :)
import mlxtend


from model import FashionMNISTModelV2

BATCH_SIZE = 16
EPOCHS = 7
torch.manual_seed(42)

# Writing device agnostic-code:
device = "cuda" if torch.cuda.is_available() else "cpu"


# getting the dataset
print("Downloading the dataset\n")
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

class_names = train_data.classes

# preparing a datat loader

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

# Creating an instance of our model
model_2 =FashionMNISTModelV2(input_shape=1,
                             hidden_units=32,
                             output_shape=len(class_names)).to(device)

# Choosing a loss function and an optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)


from helper_fn import train_step, test_step,accuracy_fn,print_train_time
train_time_start_model2 = timer()
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
epochs = EPOCHS

for epoch in tqdm(range(epochs)):
    tqdm.write(f"\nEpoch: {epoch}\n----------")

    train_loss, train_acc = train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)

    test_loss, test_acc = test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

train_time_end_model2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model2,
                                            end=train_time_end_model2,
                                            device=device)

test_samples = []
test_labels = []
for sample , lable in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(lable)

from helper_fn import make_predictions

pred_probs = make_predictions(model=model_2,
                              data=test_samples)

# Turn the prediction probabilities into prediction labels by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)

# Visualising random otput sample


plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):

    plt.subplot(nrows, ncols, i+1)

    plt.imshow(sample.squeeze(), cmap="gray")

    pred_label = class_names[pred_classes[i]]

    truth_label = class_names[test_labels[i]]

    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")
        plt.axis(False)
    else:
        plt.title(title_text, fontsize=10, c="r")
        plt.axis(False)
plt.axis(False)

#----Plotting a confusion matrix and curves for evaluation
y_preds = []
model_2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    # Forward pass
    y_logit = model_2(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)


from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)

# Create a figure with 2 subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


# ----
ax1.plot(train_losses, marker="o", label="Train Loss")
ax1.plot(test_losses, marker="o", label="Test Loss")
ax1.set_title("Training & Test Loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

# ----
ax2.plot(train_accuracies, marker="o", label="Train Accuracy")
ax2.plot(test_accuracies, marker="o", label="Test Accuracy")
ax2.set_title("Training & Test Accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

plt.tight_layout()
plt.show()

# Saving the trained model
from pathlib import Path
# 1. Create models directory
MODEL_PATH = Path("Models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_FASHIONMNIST_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
