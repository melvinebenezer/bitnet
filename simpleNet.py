import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import os
from torchao.prototype.dtypes.bitnet import BitnetTensor

_ = torch.manual_seed(0)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
# Load the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
# Load the MNIST test set
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, device=device, bias=bias)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

class BitLinearTrain(nn.Linear):
    def forward(self, x):
        w = self.weight
        x_norm = x
        x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

    def activation_quant(self, x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        return y

    def weight_quant(self, w):
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u

class VerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(VerySimpleNet, self).__init__()
        self.linear1 = BitLinearTrain(28*28, hidden_size_1)
        self.linear2 = BitLinearTrain(hidden_size_1, hidden_size_2)
        self.linear3 = BitLinearTrain(hidden_size_2, 8)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def quantize_func(linear):
    new_linear = BitLinear(linear.in_features, linear.out_features, device=linear.weight.device, bias=None)
    new_linear.weight = torch.nn.Parameter(BitnetTensor.from_float(linear.weight.t()), requires_grad=False)
    del(linear)
    return new_linear

def swap_linear_layers(
    module: nn.Module,
    from_float_func,
) -> nn.Module:
    def replace_linear(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                new_module = from_float_func(child)
                setattr(module, name, new_module)
            else:
                replace_linear(child)
    replace_linear(module)
    # import gc
    # torch.cuda.empty_cache()
    # gc.collect()
    # torch.cuda.synchronize()
    return module

def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()
        loss_sum = 0
        num_iterations = 0
        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y  =  data
            x = x.to(device)
            y = y % 8 # Limit the number of classes to 8
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

def test(model: nn.Module, total_iterations: int = None):
    correct = 0
    total = 0
    iterations = 0
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x.view(-1, 28*28))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
            iterations += 1
            if total_iterations is not None and iterations >= total_iterations:
                break

    print("Accuracy: ", {round(correct/total, 3)})
    

if __name__ == "__main__":
    # Define the device 
    device = "cpu"

    net = VerySimpleNet().to(device)

    MODEL_FILENAME = "simplenet_full.pth"
    BITNET_MODEL_FILENAME = "simplenet_bitnet.pth"

    if Path(MODEL_FILENAME).exists():
        net.load_state_dict(torch.load(MODEL_FILENAME))
    else:
        train(train_loader, net, epochs=5)
    torch.save(net.state_dict(), MODEL_FILENAME)
    test(net)
    print("Quantizing the model")
    swap_linear_layers(net, quantize_func)
    torch.save(net, BITNET_MODEL_FILENAME)  
    test(net)
