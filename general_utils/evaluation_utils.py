from typing import Callable
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import torch.nn.functional as F

# Constants
DEFAULT_BATCH_SIZE = 32


class EvaluationUtils(object):
    @staticmethod
    def eval_two_models(model1: Module, model2: Module, dataset: Dataset,
                        combine_func: Callable,
                        accuracy_func: Callable, batch_size: DEFAULT_BATCH_SIZE,
                        device: torch.device) -> float:
        validation_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model1.eval()
        model2.eval()
        total_num_images = 0
        aggregate_accuracy = 0.0
        for batch_num, (images, labels) in enumerate(validation_loader):
            num_images_in_batch = images.shape[0]
            total_num_images += num_images_in_batch
            images = images.to(device)
            labels = labels.to(device)
            model1_output = model1(images)
            model2_output = model2(images)
            combined_output = combine_func(model1_output, model2_output)
            current_accuracy = accuracy_func(combined_output, labels)
            aggregate_accuracy += current_accuracy * num_images_in_batch

        final_accuracy = aggregate_accuracy / total_num_images
        return final_accuracy

# Code to evaluate eval_two_models func


class RandomDataSet(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, index) -> T_co:
        rand_tensor = torch.randn(7, 18) + 10
        rand_tensor[:4, 9:] = 0
        rand_label = rand_tensor.argmax(dim=1)
        # if index % 10 == 0:
        #     rand_label[0] += 1
        return rand_tensor, rand_label


class Model1(Module):
    @staticmethod
    def forward(x):
        x = x.clone()
        x[:, 4:, :] = 0
        return x[:, :, :9]


class Model2(Module):
    @staticmethod
    def forward(x):
        x = x.clone()
        x[:, :4, :] = 0
        return x


def label_accuracy(output: torch.tensor, labels: torch.tensor) -> float:
    output_labels = output.argmax(dim=2)
    num_labels = labels.shape[0]
    num_correct_labels = output_labels.eq(labels).all(dim=1).sum().item()
    return num_correct_labels / num_labels


def combine_outputs(output1: torch.tensor, output2: torch.tensor) -> torch.tensor:
    output1_shape = output1.shape
    output2_shape = output2.shape
    padding = output2_shape[2] - output1_shape[2]
    output1_padded = F.pad(input=output1, pad=(0, padding), mode='constant', value=0)
    return output1_padded + output2


def two_model_accuracy(output1: torch.tensor, output2: torch.tensor, label: torch.tensor) -> float:
    combined_output = combine_outputs(output1, output2)
    return label_accuracy(combined_output, label)


if __name__ == '__main__':
    random_dataset = RandomDataSet()
    first_model = Model1()
    second_model = Model2()
    accuracy = EvaluationUtils.eval_two_models(first_model, second_model, dataset=random_dataset,
                                               combine_func=combine_outputs,
                                               accuracy_func=label_accuracy,
                                               batch_size=32,
                                               device=torch.device('cpu'))
    print(f"final accuracy {accuracy}")
