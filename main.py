# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt

# -----------------------------
# 1. Data Transformations
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomRotation((-15.0, 15.0), fill=(0,)),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------------
# 2. Dataset & Dataloader
# -----------------------------
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)



# %%
# -----------------------------
# 3. Model Definition (<20k params)
# -----------------------------
# Model definition (parameter-efficient, <20k params)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False), # -> 4x28x28
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, padding=1, bias=False), # -> 8x28x28
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # -> 8x14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False), # -> 16x14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), # -> 32x14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # -> 32x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 40, 3, padding=1, bias=False), # -> 40x7x7
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1) # -> 40x1x1
        self.fc = nn.Linear(40, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# %%
# -----------------------------
# 4. Training & Testing Functions
# -----------------------------
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)
    train_loss, correct = 0, 0

    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        pbar.set_description(desc=f"Train Loss={loss.item():.4f} Accuracy={100. * correct / len(train_loader.dataset):.2f}")

    return 100. * correct / len(train_loader.dataset)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n')
    return acc


# %%
# -----------------------------
# 5. Setup Training
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

# %%
# -----------------------------
# 6. Run Training
# -----------------------------
epochs = 20
train_acc_list, test_acc_list = [], []

for epoch in range(1, epochs+1):
    print(f"Epoch {epoch}")
    train_acc = train(model, device, train_loader, optimizer, criterion)
    test_acc = test(model, device, test_loader, criterion)
    scheduler.step()
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

# %%
# -----------------------------
# 7. Plot Results
# -----------------------------
plt.plot(train_acc_list, label='Train Acc')
plt.plot(test_acc_list, label='Test Acc')
plt.legend()
plt.title("Training vs Test Accuracy")
plt.show()



# %%
# -----------------------------
# 8. Model Architecture Checks
# -----------------------------
def model_checks(model):
    print('--- Model Architecture Checks ---')
    # Total Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameter Count in Model: {total_params}\n')
    print('Layer-wise Parameter Details (in model order):')
    print('-'*80)
    # Get layers in order as defined in model
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            print(f"\nBlock: {name} (nn.Sequential)")
            for subname, submodule in module.named_children():
                layer_name = f'{name}.{subname} ({submodule.__class__.__name__})'
                layer_params = sum(p.numel() for p in submodule.parameters())
                details = ''
                if isinstance(submodule, nn.Conv2d):
                    details = f'Convolution: {submodule.in_channels} input channels, {submodule.out_channels} output channels, kernel size {submodule.kernel_size}, bias {submodule.bias is not None}'
                elif isinstance(submodule, nn.BatchNorm2d):
                    details = f'BatchNorm: {submodule.num_features} features, affine {submodule.affine}'
                elif isinstance(submodule, nn.ReLU):
                    details = 'Activation: ReLU (no parameters)'
                elif isinstance(submodule, nn.MaxPool2d):
                    details = f'MaxPooling: kernel size {submodule.kernel_size}, stride {submodule.stride}'
                elif isinstance(submodule, nn.Dropout):
                    details = f'Dropout: probability {submodule.p}'
                print(f'  {layer_name:40} | Params: {layer_params:6d} | {details}')
        else:
            layer_name = f'{name} ({module.__class__.__name__})'
            layer_params = sum(p.numel() for p in module.parameters())
            details = ''
            if isinstance(module, nn.AdaptiveAvgPool2d):
                details = f'Global Average Pooling: output size {module.output_size} (no parameters)'
            elif isinstance(module, nn.Linear):
                details = f'Fully Connected: {module.in_features} input features, {module.out_features} output features, bias {module.bias is not None}'
            print(f'  {layer_name:40} | Params: {layer_params:6d} | {details}')
    print('-'*80)
    print('\nSummary:')
    # BatchNorm
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    print(f'BatchNorm2d layers used: {len(bn_layers)}')
    # Dropout
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    print(f'Dropout layers used: {len(dropout_layers)}')
    # Fully Connected & GAP
    fc_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    gap_layers = [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    print(f'Fully Connected (Linear) layers used: {len(fc_layers)}')
    print(f'Global Average Pooling layers used: {len(gap_layers)}')
    print('---------------------------------')

# Run checks on current model
model_checks(model)


