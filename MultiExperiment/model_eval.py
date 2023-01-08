from typing import List, Tuple, Union, Any
import torch
import math


class RecognitionModelEvaluatorException(Exception):
    pass


class RecognitionModelEvaluator(object):

    @staticmethod
    def edit_distance_accuracy(model_output: torch.Tensor, gt: torch.tensor, max_allowed_dist: int = 1) -> float:
        if not model_output.dim() - 1 == gt.dim():
            raise RecognitionModelEvaluatorException(f"model_output must be 1 dim bigger then ground_truth. "
                                                     f"got: model_output: {model_output.dim()}, gt: {gt.dim()}")
        if gt.dim() == 1:  # Single label accuracy
            model_output = model_output.unsqueeze(0)
            gt = gt.unsqueeze(0)
        if gt.dim() != 2:
            raise RecognitionModelEvaluatorException(f"Error in input dimensions "
                                                     f"got: model_output: {model_output.dim()}, gt: {gt.dim()}")
        model_output_int_seq = model_output.argmax(dim=1)
        output_gt_compare = torch.eq(model_output_int_seq, gt)
        num_mistakes_per_label = output_gt_compare.shape[1] - output_gt_compare.sum(dim=1)
        is_label_correct = num_mistakes_per_label <= max_allowed_dist
        return float(is_label_correct.sum() / is_label_correct.numel())

    @staticmethod
    def _recognition_output_get_word_prob(char_prob: torch.Tensor,
                                          word: torch.Tensor) -> float:
        if word.dim() != 1:
            raise RecognitionModelEvaluatorException(f"word must be of dim 1 got {word.dim()}")
        if word.shape[0] != char_prob.shape[0]:
            raise RecognitionModelEvaluatorException(f"word and recognition_model_output must be of same length "
                                                     f"got word len:{word.shape[0]}, "
                                                     f"model_output len:{char_prob.shape[0]}")
        word_char_probs = char_prob.gather(dim=1, index=word.unsqueeze(1)).squeeze(1)
        word_prob = word_char_probs.cumprod(dim=0)[-1]
        return float(word_prob)

    @staticmethod
    def _recognition_output_get_best_k_results(char_prob: torch.Tensor,
                                               sequence_dim: int = 0,
                                               character_dim: int = 1,
                                               k: int = 1) -> List[List[Union[Any, float]]]:
        """
        Assumptions:
            sequence_dim: int = 0,
            character_dim: int = 1,
        """
        sequence_len = char_prob.shape[sequence_dim]
        take_n_best_of_each_char = math.ceil(k ** (1 / sequence_len))
        char_indexs_sorted_by_prob = char_prob.argsort(dim=character_dim, descending=True)
        char_indxes_of_best_prob = char_indexs_sorted_by_prob[:, :take_n_best_of_each_char]
        char_indxes_all_permutations = torch.cartesian_prod(*[char_indxes_of_best_prob[x]
                                                              for x in range(char_indxes_of_best_prob.shape[0])])
        permutation_prob_list = []
        for permutation in char_indxes_all_permutations:
            if len(permutation_prob_list) > k:
                break
            permutation_prob = RecognitionModelEvaluator._recognition_output_get_word_prob(char_prob, permutation)
            permutation_prob_list.append([permutation, permutation_prob])
        permutation_prob_list.sort(key=lambda x: x[1])
        return permutation_prob_list

    @staticmethod
    def recognition_pr_curve(recognition_model_output: torch.Tensor, label: torch.Tensor, max_k: int) -> \
            Tuple[List[float], List[float]]:
        for k in range(1, max_k):
            pass


def test_edit_distance_1_accuracy() -> None:
    a = torch.randn(32, 18, 7)
    b = a.argmax(dim=1)
    assert RecognitionModelEvaluator.edit_distance_accuracy(a, b) == 1
    assert RecognitionModelEvaluator.edit_distance_accuracy(a[0], b[0]) == 1
    b[0, 1] -= 1
    assert RecognitionModelEvaluator.edit_distance_accuracy(a, b) == 1
    assert RecognitionModelEvaluator.edit_distance_accuracy(a[0], b[0]) == 1
    b[0, 2] -= 1
    assert RecognitionModelEvaluator.edit_distance_accuracy(a, b) == 1 - 1 / 32
    assert RecognitionModelEvaluator.edit_distance_accuracy(a[0], b[0]) == 0
    b[0, 3] -= 1
    b[1, 0] -= 1
    assert RecognitionModelEvaluator.edit_distance_accuracy(a, b) == 1 - 1 / 32
    b[1, 1] -= 1
    assert RecognitionModelEvaluator.edit_distance_accuracy(a, b) == 1 - 2 / 32
    b[:, 4] -= 1
    assert RecognitionModelEvaluator.edit_distance_accuracy(a, b) == 1 - 2 / 32
    assert RecognitionModelEvaluator.edit_distance_accuracy(a, b, max_allowed_dist=0) == 0


if __name__ == '__main__':
    test_edit_distance_1_accuracy()
