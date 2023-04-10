
from datasets import Datasets
from logistic_regression import LogReg
from model_evaluation import Evaluation
import numpy as np

pth = "../"
data = Datasets(pth + "X_train.csv", pth + "X_test.csv", pth + "X_val.csv",
                pth + "y_train.csv", pth + "y_test.csv", pth + "y_val.csv")
log = LogReg(data)

# One vs Rest Logistic regression prediction
log_y_pred = log.one_v_rest()
eva = Evaluation()
print(eva.accuracy(log_y_pred, data.y_test))
print(np.count_nonzero(data.y_test == "Fatal"))
eva_conf = eva.confusion(log_y_pred, data.y_test)
eva.plot_confusion(eva_conf)

ecoc_y_pred = log.ecoc()
print(eva.accuracy(ecoc_y_pred, data.y_test))
eva_conf = eva.confusion(ecoc_y_pred, data.y_test)
eva.plot_confusion(eva_conf)
