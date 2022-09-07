from typing import List, Tuple
import torch
import math


class RecognitionModelEvaluatorException(Exception):
    pass


class RecognitionModelEvaluator(object):
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
                                               k: int = 1) -> List[List[torch.Tensor, float]]:
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
        return permutation_prob_list

    @staticmethod
    def recognition_pr_curve(recognition_model_output: torch.Tensor, label: torch.Tensor, max_k: int) -> Tuple[List[float], List[float]]:
        for k in range(1, max_k):
            pass


if __name__ == '__main__':
    pass
