import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

def make_matrix(TP,TN, FP, FN):
    multiclass = np.array([[TP, FP],
                           [FN, TN]]
                            )

    class_names = ['Malignant', 'Beningn']

    fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names=class_names)
    plt.show()