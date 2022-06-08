from time import sleep
from typing import List, Tuple
from pathlib import Path
from tensorboard import program
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Constants
RUNS_PATH = Path(r"C:\Users\צבי\Desktop\עבודה צבי\LPRIL\KMUtils\GeneralUtils\runs")

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def exp(exp_name: str, train_dataloader: DataLoader, test_dataloader: DataLoader,
        lr: int) -> Tuple[float, float]:
    running_loss = 0.0
    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar('Training/loss',
                                  running_loss / 1000,
                                  epoch * len(trainloader) + i)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(net, inputs, labels),
                                  global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
    print('Finished Training')


def multi_exp(multi_exp_name: str, model, train_dataset: Dataset, test_dataset: Dataset, lr_list: List[float],
              batch_size_list: List[int], run_tensorboard: bool):
    multi_exp_results_path = RUNS_PATH.joinpath(multi_exp_name)
    if run_tensorboard:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', multi_exp_results_path])
        tb.launch()
    writer = SummaryWriter(multi_exp_results_path.joinpath('MainMultiExp'))
    writer.add_graph(model, images)
    writer.add_image('train_dataset',
    for batch_size in batch_size_list:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        for lr in lr_list:
            exp_name = f"batch_size {batch_size}"
            exp()


if __name__ == '__main__':
    # Init
    trainset = torchvision.datasets.FashionMNIST('./data',
                                                 download=True,
                                                 train=True,
                                                 transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
                                                download=True,
                                                train=False,
                                                transform=transform)



    writer = SummaryWriter(r'runs/Exp1')
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # # images, labels = next(iter(trainloader))
    # # ...log the running loss
    # for n_iter in range(200):
    #     writer.add_scalar('Loss/train', np.random.random(), n_iter)
    #     writer.add_scalar('Loss/test', np.random.random(), n_iter)
    #     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    #     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    #     sleep(1)
    #     print(f"add scalar {n_iter}")

    writer = SummaryWriter(r'runs/Exp2')
    for i in range(5):
        writer.add_hparams({'lr': 0.1 * i, 'bsize': i},
                      {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i})




    # # create grid of images
    # img_grid = torchvision.utils.make_grid(images)
    #
    # # show images
    # matplotlib_imshow(img_grid, one_channel=True)
    #
    # # write to tensorboard
    # writer.add_image('four_fashion_mnist_images', img_grid)
    # writer.add_graph(net, images)
    # writer.close()
    #
    # # select random images and their target indices
    # images, labels = select_n_random(trainset.data, trainset.targets)
    #
    # # get the class labels for each image
    # class_labels = [classes[lab] for lab in labels]
    #
    # # log embeddings
    # features = images.view(-1, 28 * 28)
    # writer.add_embedding(features,
    #                      metadata=class_labels,
    #                      label_img=images.unsqueeze(1))
    # writer.close()

    # def images_to_probs(net, images):
    #     '''
    #     Generates predictions and corresponding probabilities from a trained
    #     network and a list of images
    #     '''
    #     output = net(images)
    #     # convert output probabilities to predicted class
    #     _, preds_tensor = torch.max(output, 1)
    #     preds = np.squeeze(preds_tensor.numpy())
    #     return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
    #
    # def plot_classes_preds(net, images, labels):
    #     '''
    #     Generates matplotlib Figure using a trained network, along with images
    #     and labels from a batch, that shows the network's top prediction along
    #     with its probability, alongside the actual label, coloring this
    #     information based on whether the prediction was correct or not.
    #     Uses the "images_to_probs" function.
    #     '''
    #     preds, probs = images_to_probs(net, images)
    #     # plot the images in the batch, along with predicted and true labels
    #     fig = plt.figure(figsize=(12, 48))
    #     for idx in np.arange(4):
    #         ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
    #         matplotlib_imshow(images[idx], one_channel=True)
    #         ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
    #             classes[preds[idx]],
    #             probs[idx] * 100.0,
    #             classes[labels[idx]]),
    #             color=("green" if preds[idx] == labels[idx].item() else "red"))
    #     return fig
    #
    #

