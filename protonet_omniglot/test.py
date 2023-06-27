import os
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


def test(args):
    logger.warning('Test')

    dataset = omniglot(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=False,
                       test_shots=15,
                       meta_test=True,
                       download=False)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers)

    model = PrototypicalNetwork(1,
                                args.embedding_size,
                                hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.eval()  # Set the model to evaluation mode
    
    # Load the trained model
    filename = os.path.join(args.output_folder, 'protonet_omniglot_'
                            '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))
    with open(filename, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)

    # Testing loop
    overall_accuracy = torch.tensor(0., device=args.device)
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(train_embeddings, train_targets,
                dataset.num_classes_per_task)

            with torch.no_grad():
                accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                overall_accuracy += accuracy
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

                
            if batch_idx >= args.num_batches:
                break
    
    overall_accuracy.div_(args.num_batches)
    print('Test accuracy: {:.4f}'.format(overall_accuracy.item()))
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')

    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    test(args)
