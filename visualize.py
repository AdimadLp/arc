import torch
import matplotlib.pyplot as plt
import json
import os
import test_gpt2

def visualize(model_path, data, iteration=1):
    print(f"Visualizing example {iteration} of model {model_path}")
    os.makedirs(f'plots/{model_path}', exist_ok=True)

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

    # Display the output heatmap 
    plt.savefig(f'plots/{model_path}/test_{iteration}.png')

if __name__ == '__main__':
    model_path = 'arc_model_2e-05_70'
    device = torch.device("cuda")
    model = test_gpt2.load_model(model_path).to(device)
    tokenizer = test_gpt2.load_tokenizer(model_path)

    result = test_gpt2.test_model(model, tokenizer, device)
    # Parse the JSON string to a Python list
    data = json.loads(result)
    visualize(model_path, data)