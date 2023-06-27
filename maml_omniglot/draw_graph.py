import numpy as np
import os
import matplotlib.pyplot as plt

def draw_graph(args):
    num_shots = [1, 3]
    num_ways = [5]

    # Initialize a list to store the accuracy values for each line

    for num_shot in num_shots:
        accuracy_lines = []
        for num_way in num_ways:
            # Load the accuracy values from the file
            maml_accuracy = np.load(os.path.join(args.output_folder, 'maml_omniglot_{0}shot_{1}way.npy'.format(num_shot, num_way)))
            proto_accuracy = np.load(os.path.join(args.output_folder, 'protonet_omniglot_{0}shot_{1}way.npy'.format(num_shot, num_way)))
            accuracy_lines.append(maml_accuracy)
            accuracy_lines.append(proto_accuracy)

            # Create a figure and axes for the graph
            fig, ax = plt.subplots()

            # Plot each line with a different color and label
            colors = ['r', 'g']
            labels = ['MAML', 'ProtoNet']
            for i, accuracy in enumerate(accuracy_lines):
                x = np.arange(1, len(accuracy) + 1)  # Batches from 1 to 100
                y = accuracy  # Accuracy values
                ax.plot(x, y, color=colors[i], label=labels[i])

                # Find the index of the highest accuracy value
                max_index = np.argmax(accuracy)
                max_accuracy = accuracy[max_index]
                
                # Add a marker at the highest accuracy point
                ax.plot(max_index + 1, max_accuracy, marker='o', markersize=5, color=colors[i])

                # Add a text label for the highest accuracy value
                ax.text(max_index + 1, max_accuracy, f'{max_accuracy:.2f}', ha='center', va='bottom')

            # Set the x-axis and y-axis labels
            ax.set_xlabel('Number of Batches')
            ax.set_ylabel('Accuracy')
            ax.set_title('{0}Ways_{1}Shots'.format(num_way, num_shot))

            # Add a legend to the graph
            ax.legend()

            # Display the graph
            plt.show()

   


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Draw')

    parser.add_argument('--output-folder', type=str,
                        help='Path to the output folder that contain the accuracy of saved model.')

    args = parser.parse_args()
    draw_graph(args)
