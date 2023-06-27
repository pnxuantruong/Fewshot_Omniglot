import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy

logger = logging.getLogger(__name__)


def test(args):
    dataset = omniglot(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=False,
                       test_shots=15,
                       meta_test=True,
                       download=False)  # Set download to False for testing
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(1,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.eval()  # Set the model to evaluation mode
    
    filename = os.path.join(args.output_folder, 'maml_omniglot_'
                                        '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))

    # Testing loop
    overall_accuracy = torch.tensor(0., device=args.device)
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            accuracy = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):

                # Load the trained model
                with open(filename, 'rb') as f:
                    model.load_state_dict(torch.load(f))

                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)

                test_logit = model(test_input, params=params)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            accuracy.div_(args.batch_size)
            overall_accuracy += accuracy

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_batches:
                break
    
    overall_accuracy.div_(args.num_batches)
    print('Test accuracy: {:.4f}'.format(overall_accuracy.item()))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Testing')

    parser.add_argument('folder', type=str,
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    
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
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the folder containing the trained model.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    test(args)
