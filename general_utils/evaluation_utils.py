from typing import Callable, Set
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import torch.nn.functional as F

from KMUtils.general_utils.dataset_utils import DatasetUtils
from KMUtils.general_utils.lp_utils import LPUtils
from KMUtils.general_utils.model_combiner import ModelCombiner

# Constants
DEFAULT_BATCH_SIZE = 32


class EvaluationUtils(object):
    @staticmethod
    def eval_model(model: Module, dataset: Dataset,
                   accuracy_func: Callable,
                   device: torch.device,
                   batch_size: int = DEFAULT_BATCH_SIZE) -> float:
        validation_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        total_num_images = 0
        aggregate_accuracy = 0.0
        for batch_num, (images, labels) in enumerate(validation_loader):
            num_images_in_batch = images.shape[0]
            total_num_images += num_images_in_batch
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            current_accuracy = accuracy_func(output, labels)
            aggregate_accuracy += current_accuracy * num_images_in_batch
        final_accuracy = aggregate_accuracy / total_num_images
        return final_accuracy

    @staticmethod
    def get_tracklet_recall(model: Module, lp_dir_path: str, label_accuracy_func: Callable,
                            device: torch.device, batch_size: int = DEFAULT_BATCH_SIZE) -> float:
        labeled_tracklet_ids_set: Set = LPUtils().get_labeled_tracklet_ids(lp_dir_path)
        tracklet_success_counter: int = 0
        for tracklet_id in labeled_tracklet_ids_set:
            tracklet_dataset: Dataset = DatasetUtils.get_tracklet_dataset(lp_dir_path, tracklet_id)
            tracklet_acc = EvaluationUtils.eval_model(model, dataset=tracklet_dataset,
                                                      accuracy_func=label_accuracy_func, device=device,
                                                      batch_size=batch_size)
            if tracklet_acc > 0.0:
                tracklet_success_counter += 1

        return tracklet_success_counter / len(labeled_tracklet_ids_set)


# Code to evaluate "eval_model" func
class RandomDataSet(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, index) -> T_co:
        rand_tensor = torch.randn(7, 18) + 10
        rand_tensor[:4, 9:] = 0
        rand_label = rand_tensor.argmax(dim=1)
        # if index % 2 == 0:
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


if __name__ == '__main__':
    random_dataset = RandomDataSet()
    first_model = Model1()
    second_model = Model2()
    combined_model = ModelCombiner(first_model, second_model, combine_outputs)
    accuracy = EvaluationUtils.eval_model(combined_model, dataset=random_dataset,
                                          accuracy_func=label_accuracy,
                                          batch_size=32, device=torch.device('cpu'))
    print(f"final accuracy of 'eval_model' {accuracy}")
