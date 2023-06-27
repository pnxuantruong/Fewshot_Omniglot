import multiprocessing
import os
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torch
from tqdm import tqdm
import logging
import numpy as np

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes

from model import PrototypicalNetwork
from utils import get_accuracy

logger = logging.getLogger(__name__)


def show(args):
    # Set up the Omniglot dataset
    dataset = omniglot(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=False,
                       test_shots=args.test_shots,
                       meta_test=True,
                       download=False)  # Set download to False for testing

    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle,
                                     num_workers=args.num_workers)

    model = PrototypicalNetwork(1,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.eval()  # Set the model to evaluation mode
    
    # Load the trained model
    filename = os.path.join('../output', 'protonet_omniglot_'
                            '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))
    with open(filename, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)

    # Get one batch of data
    dataloader_iter = iter(dataloader)

    for _ in range(args.batch_index - 1):
        next(dataloader_iter)
    # batch = next(iter(dataloader))
    batch = next(dataloader_iter)
    
    accuracy = torch.tensor(0., device=args.device)


    # Extract support and query sets
    support_images, support_labels = batch['train']
    support_images = support_images.to(device=args.device)
    support_labels = support_labels.to(device=args.device)

    query_images, query_labels = batch['test']
    query_images = query_images.to(device=args.device)
    query_labels = query_labels.to(device=args.device)
    
    # Select one task from the batch
    task_idx = max(0, min(args.task_index, args.batch_size - 1 ))

    # Extract support and query images and labels for the selected task
    support_images_task = support_images[task_idx]
    support_labels_task = support_labels[task_idx]
    query_images_task = query_images[task_idx]
    query_labels_task = query_labels[task_idx]

    # predict labels
    train_embeddings = model(support_images_task)

    test_embeddings = model(query_images_task)

    prototypes = get_prototypes(train_embeddings, support_labels_task,
                dataset.num_classes_per_task)

    with torch.no_grad():
         accuracy = get_accuracy(prototypes, test_embeddings, query_labels_task)


    # Convert the tensors to numpy arrays
    support_images_task = support_images_task.cpu().detach().numpy()
    support_labels_task = support_labels_task.cpu().detach().numpy()
    query_images_task = query_images_task.cpu().detach().numpy()
    query_labels_task = query_labels_task.cpu().detach().numpy()

    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - test_embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    predictions = predictions.cpu().detach().numpy()

    # Print the support labels
    print('Support Labels:', support_labels_task)

    # Print the query labels
    print('Query Labels:', query_labels_task)

    # Print the predict labels
    print('Predict Labels:', predictions)

    # Print the accuracy
    print('Accuracy: {:.4f}'.format(accuracy.item()))

    # Visualize the support images
    support_images_grid = make_grid(torch.Tensor(support_images_task), nrow=args.num_shots, pad_value=1)
    plt.imshow(support_images_grid.permute(1, 2, 0))
    plt.title('Support Images')
    plt.axis('off')
    plt.show()

    # Visualize the query images
    query_images_grid = make_grid(torch.Tensor(query_images_task), nrow=5, pad_value=1)
    plt.imshow(query_images_grid.permute(1, 2, 0))
    plt.title('Query Images')
    plt.axis('off')
    plt.show()
    
    
if __name__ == '__main__':
    multiprocessing.freeze_support()

    import argparse

    parser = argparse.ArgumentParser('Testing')

    parser.add_argument('folder', type=str,
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--test-shots', type=int, default=5,
                        help='Number of images per task (N in "N-way", default: 5).')
    
    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches to evaluate (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers for data loading (default: 1).')
    

    parser.add_argument('--batch-index', type=int, default=0,
                        help='Index of batch (default: 0).')
    parser.add_argument('--task-index', type=int, default=0,
                        help='Index of task in batch (default: 0).')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle data if trigger')

    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')
    
    show(args)