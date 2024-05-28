import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

def make_matrix(TP,TN, FP, FN):
    multiclass = np.array([[TP, FN],
                           [FP, TN]]
                            )

    class_names = ['M', 'B']

    fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=False,
                                    class_names=class_names)
    plt.show()

make_matrix(50, 87, 2, 4)