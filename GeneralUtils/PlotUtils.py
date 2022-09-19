from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class PlotUtils(object):

    @staticmethod
    def plot_single_roc_curve(fpr: np.ndarray, tpr: np.ndarray,
                              title: str = '', label='', color="darkorange",
                              axes: Optional[plt.Axes] = None, do_show: bool = True) -> Optional[plt.Axes]:
        """
        If you wish to add a roc curve to an existing plot -> Add axes
        """
        if not axes:
            _, axes = plt.subplots(1)
        lw = 2
        axes.plot(fpr, tpr,
                  color=color,
                  label=label,
                  lw=lw)
        axes.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        axes.axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
        axes.set_xlabel("False Positive Rate")
        axes.set_ylabel("True Positive Rate")
        axes.set_title(f"{title} - ROC")
        axes.legend(loc="lower right")
        if do_show:
            plt.show()
        return axes


if __name__ == '__main__':
    x = np.array([i / 10 for i in range(10)])
    y = x + 0.05
    y2 = x + 0.07
    y3 = x + 0.1
    ax = PlotUtils.plot_single_roc_curve(fpr=x, tpr=y, title='', label="test1", color="b", do_show=False)
    ax = PlotUtils.plot_single_roc_curve(fpr=x, tpr=y3, axes=ax, title='', label="test3", color="g", do_show=False)
    PlotUtils.plot_single_roc_curve(fpr=x, tpr=y2, axes=ax, title='just a test', label="test2", do_show=True)
