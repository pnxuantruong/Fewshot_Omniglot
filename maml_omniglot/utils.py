import torch

from collections import OrderedDict

# from sklearn.metrics import f1_score


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

# def get_f1(logits, targets):
#     _, predictions = torch.max(logits, dim=-1)
#     predictions = predictions.cpu().detach().numpy()
#     targets = targets.cpu().detach().numpy()
#     f1 = f1_score(targets, predictions, average='weighted')
#     return f1