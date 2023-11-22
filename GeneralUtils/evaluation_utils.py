from typing import Callable, Set, Tuple, List, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

from KMUtils.GeneralUtils.dataset_utils import DatasetUtils
from KMUtils.GeneralUtils.lp_utils import LPUtils
from KMUtils.GeneralUtils.model_combiner import ModelCombiner

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

    @classmethod
    def tracklet_precision_recall(cls, df: pd.DataFrame, tracklet_col: str,
                                  score_col: str, correct_tracklet: str,
                                  agg_type: str = 'max', plot_pr_curve: bool = False) -> Tuple[
        List[float], List[float], List[float]]:
        df_tracklet = cls.agg_by_tracklet_top_score(df, tracklet_col, score_col, agg_type)
        p, r, t = precision_recall_curve(df_tracklet[tracklet_col] == correct_tracklet, df_tracklet[score_col])
        if plot_pr_curve:
            _, ax = plt.subplots(figsize=(7, 8))
            disp = PrecisionRecallDisplay(precision=p, recall=r)
            disp.plot(ax=ax, name="TestPlot2")
            ax.set_xlim([-0.1, 1.05])
            ax.set_ylim([-0.1, 1.05])
            plt.show()
        return p, r, t

    @staticmethod
    def add_pr_curve_to_plot(precision: Union[np.ndarray, List], recall: Union[np.ndarray, List],
                             ax, name: str = '') -> None:
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot(ax=ax, name=name)

    @staticmethod
    def agg_by_tracklet_top_score(df: pd.DataFrame, tracklet_col: str,
                                  score_col: str, agg_type: str = 'max') -> pd.DataFrame:
        idx = df.groupby([tracklet_col])[score_col].transform(agg_type) == df['score']
        return df[idx].reset_index()


# Code to evaluate "eval_model" func
class RandomDataSet(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rand_tensor = torch.randn(7, 18) + 10
        rand_tensor[:4, 9:] = 0
        rand_label = rand_tensor.argmax(dim=1)
        # if index % 2 == 0:
        #     rand_label[0] += 1
        return rand_tensor, rand_label


class Model1(Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, 4:, :] = 0
        return x[:, :, :9]


class Model2(Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, :4, :] = 0
        return x


def label_accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    output_labels = output.argmax(dim=2)
    num_labels = labels.shape[0]
    num_correct_labels = output_labels.eq(labels).all(dim=1).sum().item()
    return float(num_correct_labels / num_labels)


def combine_outputs(output1: torch.Tensor, output2: torch.Tensor) -> torch.Tensor:
    output1_shape = output1.shape
    output2_shape = output2.shape
    padding = output2_shape[2] - output1_shape[2]
    output1_padded = F.pad(input=output1, pad=(0, padding), mode='constant', value=0)
    return output1_padded + output2


def eval_model_example():
    random_dataset = RandomDataSet()
    first_model = Model1()
    second_model = Model2()
    combined_model = ModelCombiner(first_model, second_model, combine_outputs)
    accuracy = EvaluationUtils.eval_model(combined_model, dataset=random_dataset,
                                          accuracy_func=label_accuracy,
                                          batch_size=32, device=torch.device('cpu'))
    print(f"final accuracy of 'eval_model' {accuracy}")


def precision_recall_example():
    _, ax = plt.subplots(figsize=(7, 8))
    df = pd.DataFrame({'y_true': ['a', 'a', 'b', 'b'], 'score': [0, 1, 0, 1]})
    p, r, t = EvaluationUtils.tracklet_precision_recall(df, "y_true", "score", 'a',
                                                        agg_type='max')
    EvaluationUtils.add_pr_curve_to_plot(p, r, ax, "Test1")
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"Threshholds: {t}")
    df = pd.DataFrame({'y_true': ['a', 'a', 'b', 'b'], 'score': [0.6, 0.8, 0.7, 0.1]})
    p, r, t = EvaluationUtils.tracklet_precision_recall(df, "y_true", "score", 'a',
                                                        agg_type='max')
    EvaluationUtils.add_pr_curve_to_plot(p, r, ax, "Test2")
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"Threshholds: {t}")
    ax.set_xlim([-0.1, 1.05])
    ax.set_ylim([-0.1, 1.05])
    plt.show()


if __name__ == '__main__':
    precision_recall_example()
    # eval_model_example()
