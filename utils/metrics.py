
def calc_accuracy(predictions, labels):
    tp_nn = 0
    fp_fn = 0
    for (prediction, label) in zip(predictions, labels):
        if prediction == label:
            tp_nn += 1
        else:
            fp_fn += 1
    return tp_nn / (tp_nn + fp_fn)
