import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.amp.grad_scaler")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.amp.autocast_mode")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0
print(f"Using {NUM_WORKERS} workers for data loading.")

CONFIG = {
    "epochs": 40,
    "batch_size": 64,
    "lr": 0.001,
    "max_digits": 3,
    "blank_token": 10,
    "n_classes": 11
}

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class MultiDigitVariableMNIST(Dataset):
    def __init__(self, mnist_dataset, num_samples=50000, max_digits=3, blank_token=10):
        self.mnist_dataset = mnist_dataset
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.blank_token = blank_token
        self.img_height = 28
        self.img_width = 28
        self.data_info = self._generate_info()

    def _generate_info(self):
        info = []
        for _ in range(self.num_samples):
            num_digits = np.random.randint(1, self.max_digits + 1)
            indices = [np.random.randint(0, len(self.mnist_dataset)) for _ in range(num_digits)]
            info.append(indices)
        return info

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        indices = self.data_info[idx]
        images = []
        labels = []

        for index in indices:
            img, label = self.mnist_dataset[index]
            images.append(img)
            labels.append(label)

        combined_img = torch.cat(images, dim=2)

        current_width = combined_img.shape[2]
        max_width = self.img_width * self.max_digits
        padding = (0, max_width - current_width, 0, 0)
        padding_value = (0 - 0.1307) / 0.3081
        padded_img = F.pad(combined_img, padding, "constant", padding_value)

        num_labels = len(labels)
        padded_labels = labels + [self.blank_token] * (self.max_digits - num_labels)
        label_tensor = torch.tensor(padded_labels, dtype=torch.long)

        return padded_img, label_tensor

class CNN_VariableDigit(nn.Module):
    def __init__(self, max_digits=3, n_classes=11):
        super(CNN_VariableDigit, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 7 * 21, out_features=512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.output_heads = nn.ModuleList([
            nn.Linear(512, n_classes) for _ in range(max_digits)
        ])

    def forward(self, x):
        x = self.conv_base(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        outputs = [head(x) for head in self.output_heads]
        return outputs

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=45),
    transforms.RandomInvert(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomApply([AddGaussianNoise(0., 0.15)], p=0.5),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

base_train_dataset_aug = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
base_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_val_transform)

train_indices = list(range(len(base_train_dataset_aug)))
np.random.shuffle(train_indices)
split_point = int(np.floor(0.9 * len(base_train_dataset_aug)))
train_idx, valid_idx = train_indices[:split_point], train_indices[split_point:]

base_train_subset = Subset(base_train_dataset_aug, train_idx)
base_valid_subset = Subset(base_train_dataset_aug, valid_idx)

train_dataset = MultiDigitVariableMNIST(base_train_subset, num_samples=30000, max_digits=CONFIG["max_digits"])
valid_dataset = MultiDigitVariableMNIST(base_valid_subset, num_samples=6000, max_digits=CONFIG["max_digits"])
test_dataset = MultiDigitVariableMNIST(base_test_dataset, num_samples=10000, max_digits=CONFIG["max_digits"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model = CNN_VariableDigit(max_digits=CONFIG["max_digits"], n_classes=CONFIG["n_classes"]).to(device)

try:
    model = torch.compile(model)
    print("Model compiled successfully (PyTorch 2.0+).")
except Exception:
    print("Could not compile model (PyTorch < 2.0 or other error).")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

patience = 5
epochs_no_improve = 0
best_val_loss = float('inf')
best_model_path = 'best_model_improved.pth'
manual_stop_model_path = 'manual_stop_model_improved.pth'

print("Starting Training on 1, 2, and 3-Digit sequences...")
for epoch in range(CONFIG["epochs"]):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            outputs = model(images)
            total_loss = 0
            for i in range(CONFIG["max_digits"]):
                total_loss += criterion(outputs[i], labels[:, i])
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
    epoch_train_loss = running_loss / len(train_loader)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(images)
                total_loss = 0
                for i in range(CONFIG["max_digits"]):
                    total_loss += criterion(outputs[i], labels[:, i])
            running_val_loss += total_loss.item()
    epoch_val_loss = running_val_loss / len(valid_loader)

    print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(epoch_val_loss)
    new_lr = optimizer.param_groups[0]['lr']

    if new_lr < old_lr:
        print(f"Learning rate reduced from {old_lr} to {new_lr}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(model_to_save.state_dict(), best_model_path)
        print(f"Validation loss improved. Saving best model to {best_model_path}")
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print(f"\n--- Early stopping triggered ---")
        print(f"Validation loss did not improve for {patience} consecutive epochs.")
        print(f"The best model from epoch {epoch + 1 - patience} was saved to '{best_model_path}'")
        break

    if (epoch + 1) % 5 == 0 and (epoch + 1) < CONFIG["epochs"]:
        user_input = input(f"\nFinished epoch {epoch + 1}. Continue training? (y/n): ").lower()
        if user_input not in ['y', 'yes']:
            print("Stopping training based on user input.")
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), manual_stop_model_path)
            print(f"Current model state saved to {manual_stop_model_path}")
            break

print("Training finished.")

eval_model = CNN_VariableDigit(max_digits=CONFIG["max_digits"], n_classes=CONFIG["n_classes"]).to(device)
print(f"\nLoading best model from '{best_model_path}' for final evaluation...")
eval_model.load_state_dict(torch.load(best_model_path))
eval_model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = eval_model(images)

        predicted_digits = []
        for i in range(CONFIG["max_digits"]):
            _, predicted = torch.max(outputs[i].data, 1)
            predicted_digits.append(predicted)

        predicted_sequences = torch.stack(predicted_digits, dim=1)
        correct_sequences = torch.all(predicted_sequences == labels, dim=1)
        correct += correct_sequences.sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Exact Sequence Match Accuracy on test images: {accuracy:.2f}%")

def visualize_variable_predictions():
    eval_model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    outputs = eval_model(images)
    predicted_digits = []
    for i in range(CONFIG["max_digits"]):
        _, predicted = torch.max(outputs[i].data, 1)
        predicted_digits.append(predicted)

    images = images.cpu().numpy()

    fig = plt.figure(figsize=(20, 10))
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        img = images[idx].squeeze()

        mean = 0.1307
        std = 0.3081
        img = img * std + mean
        ax.imshow(np.clip(img, 0, 1), cmap='gray')

        pred_list = [str(p[idx].item()) for p in predicted_digits if p[idx].item() != CONFIG["blank_token"]]
        pred_str = "".join(pred_list) if pred_list else "None"
 
        actual_list = [str(l.item()) for l in labels[idx] if l.item() != CONFIG["blank_token"]]
        actual_str = "".join(actual_list)

        is_correct = (pred_str == actual_str)
        ax.set_title(f"Pred: {pred_str}\nActual: {actual_str}",
                     color=("green" if is_correct else "red"), fontsize=14)
    plt.show()

visualize_variable_predictions()
