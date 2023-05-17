import numpy as np



def compute_acc(y_true, y_pred):
    # smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    acc = (y_true_f==y_pred_f).astype(np.float32).mean()
    return acc

def compute_acc_batch(y_trues, y_preds):
    assert len(y_trues) == len(y_preds)
    running_acc = []
    for i in range(len(y_trues)):
        running_acc.append(compute_acc(y_trues[i], y_preds[i]))
    return running_acc
        