import numpy as np



def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coefficient_batch(y_trues, y_preds):
    assert len(y_trues) == len(y_preds)
    dices = []
    for i in range(len(y_trues)):
        dices.append(dice_coefficient(y_trues[i], y_preds[i]))
    return dices
        