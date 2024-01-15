import torch
import matplotlib.pyplot as plt
import json
import os
import test_gpt2

def heatmap(graph_name, data, temperature, iteration=1):
    print(f"Visualizing example {iteration} of model {graph_name} with temperature {temperature}")

    figure, axis = plt.subplots(len(data), 2)
    # Iterate over each item in the data list
    for i in range(0, len(data)):

        # Get the input and output data
        input = data[i]['input']
        output = data[i]['output']

        # Create a heatmap of the input data
        axis[i, 0].imshow(input, cmap='rainbow')

        # Hide x-axis and y-axis ticks
        axis[i, 0].set_xticks([])
        axis[i, 0].set_yticks([])

        # Create a heatmap of the output data
        axis[i, 1].imshow(output, cmap='rainbow')

        # Hide x-axis and y-axis ticks
        axis[i, 1].set_xticks([])
        axis[i, 1].set_yticks([])

    # Save the output heatmap
    os.makedirs(f'plots/{graph_name}', exist_ok=True)
    plt.savefig(f'plots/{graph_name}/test_{temperature}_{iteration}.png')
    
    # Close the figure
    plt.close(figure)

def graph(model_name, learning_rate):
    print(f"Visualizing stats of model {model_name}")
    with open(f'stats/{model_name}_{learning_rate}.json', 'r') as file:
        stats = json.load(file)
    # Create a figure
    figure, axis = plt.subplots(1, 2)

    # Create a list of epochs
    epochs = [stat['epoch'] for stat in stats]
    # Create a list of correct size examples
    correct_size_examples = [stat['correct_size_examples'] for stat in stats]
    # Create a list of visualizable examples
    visualizable_examples = [stat['visualizable_examples'] for stat in stats]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Plot the data for correct size examples
    ax.plot(epochs, correct_size_examples, color='blue', label='Correct size examples')

    # Plot the data for visualizable examples
    ax.plot(epochs, visualizable_examples, color='red', label='Visualizable examples')

    # Set the title
    ax.set_title("Correct Size vs Visualizable Examples")

    # Set the x-axis label
    ax.set_xlabel("Epoch")

    # Set the y-axis label
    ax.set_ylabel("Examples")

    # Add a legend
    ax.legend()

    # Save the figure
    os.makedirs(f'stats', exist_ok=True)
    plt.savefig(f'stats/{model_name}_{learning_rate}.png')

    # Close the figure
    plt.close(fig)

if __name__ == '__main__':
    model_path = 'gpt2'
    learning_rate = 2e-05
    graph(model_path, learning_rate)

"""
    top_k = 30
    for k in range(1, 6+1):
        top_p = 0.90
        for l in range(1, 5+1):
            for i in range(1, 5+1):
                        result = test_gpt2.test_model(model, tokenizer, device, top_k=top_k, top_p=top_p)
                        # Check if the result is not empty
                        if not result:
                            continue
                        # Parse the JSON string to a Python list
                        data = json.loads(result)
                        visualize(model_path,data, i, top_k=top_k, top_p=top_p)
            top_p += 0.01
        top_k += 5
"""