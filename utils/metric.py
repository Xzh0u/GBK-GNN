def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def compute_correct_num(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def compute_sigma_acc(sigma, ground_truth, bound=0.1):
    count = 0
    assert len(sigma) == len(ground_truth)
    for i in range(len(sigma)):
        if sigma[i] - ground_truth[i] <= bound:
            count += 1

    return count / len(sigma)
