from typing import Callable

from torch.nn import Module


class ModelCombiner(Module):
    def __init__(self, model1: Module, model2: Module, combine_func: Callable) -> None:
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.combine_func = combine_func

    def forward(self, x):
        model1_results = self.model1(x)
        model2_results = self.model2(x)
        combined_result = self.combine_func(model1_results, model2_results)
        return combined_result
