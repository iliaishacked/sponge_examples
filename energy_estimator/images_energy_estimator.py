import torch
import argparse
import numpy as np
from torch import nn
from torch import optim
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from collections import defaultdict
import time
import pickle

import energy_estimator.analyse as simul


class Net(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(Net, self).__init__()
        ic, ih, iw = input_shape
        self.conv1 = nn.Conv2d(ic, 256, 3, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        return self.fc2(x)


def get_trainable_image(image):
    # take original image in numpy format, this image should have been preprocessed
    tensor_image = torch.from_numpy(image).float()
    tensor_image = torch.nn.Parameter(tensor_image, requires_grad=True)
    return tensor_image


def compute_loss(output, target):
    return torch.sum(torch.abs(output - target))


def compute_loss_no_abs(output, target):
    return torch.sum(output - target)


def renorm(image, min_value=0.0, max_value=1.0):
    return torch.clamp(image, min_value, max_value)

def score_me(datas, model, hardware, hardware_worst, stats):

    reses = []

    hooks = simul.add_hooks(model, stats)

    for i, dat in enumerate(datas):
        stats.__reset__()
        _ = model(dat.unsqueeze(dim=0))
        energy_est = simul.get_energy_estimate(stats, hardware)
        energy_est_worst = simul.get_energy_estimate(stats, hardware_worst)
        rs = energy_est/energy_est_worst
        reses.append(rs)
        print(f"{i} {rs}", end="\r")
    print()

    simul.remove_hooks(hooks)

    return reses

def build_dataset(
        train_data_x=None, train_data_y=None,
        test_data_x=None, test_data_y=None, random=False, model=None,
        random_shape=(1, 28, 28), savename=None, gpu=None):
    # better provide train_data_x and test_data_x as post-transformation images

    if random:
        hardware = simul.HardwareModel(optim=True)
        hardware_worst = simul.HardwareModel(optim=False)
        stats = simul.StatsRecorder()

        train_data_x = torch.Tensor(np.random.rand(5000, *random_shape))
        test_data_x  = torch.Tensor(np.random.rand(100, *random_shape))

        train_data_y = torch.Tensor(np.random.rand(5000, 1))
        test_data_y  = torch.Tensor(np.random.rand(100, 1))

        if gpu is not None:
            train_data_x = train_data_x.to(gpu)
            test_data_x = train_data_x.to(gpu)

        if model is not None:
            with torch.no_grad():
                train_data_y = score_me(train_data_x, model, hardware,
                        hardware_worst, stats)
                test_data_y  = score_me(test_data_x, model, hardware,
                        hardware_worst, stats)
            print()

            train_data_y = torch.Tensor(train_data_y)
            test_data_y = torch.Tensor(test_data_y)

        if savename is not None:
            print(f"Saving to {savename} ... ")
            pickle.dump([train_data_x, train_data_y, test_data_x, test_data_y],
                    open(savename, "wb" ))

    # train loader
    dataset = data.TensorDataset(train_data_x, train_data_y)
    train_dataloader = data.DataLoader(dataset)

    # test loader
    dataset = data.TensorDataset(test_data_x, test_data_y)
    test_dataloader = data.DataLoader(dataset)
    return (train_dataloader, test_dataloader)


def build_adversarial_image(
        image, label, model, iterations=10, alpha=0.01, random=False):
    if random:
        image = np.random.rand(1, 1, 28, 28)
        label = torch.Tensor(np.random.rand(1))
    model.eval()
    numpy_image = image
    for i in range(iterations):
        tensor_image = get_trainable_image(numpy_image)
        tensor_image.grad = None
        pred = model(tensor_image)
        loss_with_sign = compute_loss_no_abs(pred, label)
        loss_with_sign.backward()
        # ascending on gradients
        adv_noise = alpha * tensor_image.grad.data
        tensor_image = tensor_image - adv_noise
        # renorm input
        tensor_image = renorm(tensor_image)
        numpy_image = tensor_image.detach().numpy()
    return image, tensor_image



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = compute_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end="\r")
    print()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += compute_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \n'.format(test_loss))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Energy Estimator')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Loading the current Model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")


    model = Net(
        (1, 28, 28),
        1,
    ).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    train_loader, test_loader = build_dataset(random=True, model=model)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # testing enabled
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    # testing enabled
    org_image, adv_image = build_adversarial_image(None, None, model, random=True)

    model.eval()
    before = time.time()
    res = model(org_image)
    after = time.time()

    if args.save_model:
        torch.save(model.state_dict(), "checkpoints/image_energy_est.pt")


if __name__ == '__main__':
    main()
