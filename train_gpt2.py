"""
This script trains a GPT-2 model on the ARC (Abstraction and Reasoning Corpus) dataset.
It loads the ARC dataset, tokenizes the input data, and trains the model using the AdamW optimizer.
After each epoch, it evaluates the model by generating tests and visualizing the results.
The trained model and tokenizer are saved, and the training statistics are recorded.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
import test_gpt2
import visualize
import argparse

# Function to load ARC tasks from a directory
def load_arc_data(directory):
    tasks = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                tasks.append(json.load(file))
    return tasks

# Function to convert a task to a string
def task_to_string(task):
    pairs = task['train'] + task.get('test', [])
    return '\n'.join(f"Input: {pair['input']} Output: {pair['output']}" for pair in pairs)

# Class to prepare ARC tasks for training
class ArcDataset(Dataset):
    def __init__(self, json_data, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        for task in json_data:
            task_str = task_to_string(task)
            tokenized_data = tokenizer(task_str, return_tensors="pt", padding='max_length', truncation=True, max_length=tokenizer.model_max_length)
            self.data.append((tokenized_data['input_ids'], tokenized_data['attention_mask']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Function to train a GPT-2 model on the ARC dataset
def train(model_path, learning_rate, batch_size, model_name=None, stats=[]):
    # Set model_name to model_path if the training is not a continuation of a previous training
    if model_name is None:
        model_name = model_path

    # If the training is a continuation of a previous training, load the stats
    if stats == []:
        epoch = 0
        max_correct_size_tests = 0
    else:
        # Get the last epoch from stats
        epoch = stats[-1]['epoch']
        # Get the max correct size tests from stats
        max_correct_size_tests = max([stat['correct_size_tests'] for stat in stats])
    
    print(f"model_path: {model_path}\nlearning_rate: {learning_rate}\nbatch_size: {batch_size}\nmodel_name: {model_name}\nepoch: {epoch}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Set the padding token to be the same as the end-of-sentence (EOS) token for the tokenizer.
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # Specify how the dataloader should combine input_ids and attention_masks into batches
    def collate_batch(batch):
        input_ids, attention_masks = zip(*batch)
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        return input_ids_padded, attention_masks_padded

    # Define the device to be used for training GPU (cuda) or CPU (cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and move it to the device
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print("Model loaded")

    # Load training data
    train_data = load_arc_data('training')
    train_dataset = ArcDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    print("Taining data loaded")

    # Load validation data
    val_data = load_arc_data('evaluation')
    val_dataset = ArcDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
    print("Evaluation data loaded")

    # Initialize optimizer that will be used to update the model parameters during training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train model
    print("Start training...")
    while True:
        # Set model to training mode
        model.train()

        # Iterate over the training data and train the model on each batch
        for step, batch in enumerate(train_loader, start=1):
            inputs, masks = batch 
            inputs, masks = inputs.to(device), masks.to(device)

            # Reset gradients at the start of each batch
            optimizer.zero_grad()
            # pass the inputs through the each layer of the model and get the outputs from the last layer
            outputs = model(input_ids=inputs, attention_mask=masks, labels=inputs)
            # Get the loss which measures how far the model predictions are from the actual values
            loss = outputs.loss
            # Calculate gradients
            loss.backward()

            # Update model parameters after each batch
            optimizer.step()  

            # Print the loss every 10 steps
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

        # Ensure any remaining gradients are pushed through at the end of each epoch
        optimizer.step()  
        optimizer.zero_grad()

        epoch += 1
        print(f"Epoch {epoch} completed with loss {loss.item()}")

        visualizable_tests = 0
        correct_size_tests = 0
        
        # Evaluate model every epoch by generating 5 tests called from test data '0c786b71.json' and visualize them
        temperature = 0.1
        for t in range(1, 10+1):
                for i in range(1, 3+1):
                            result = test_gpt2.test_model(model, tokenizer, device, temperature)

                            # Check for different result types
                            if result == 'invalid output':
                                continue
                            elif result == 'invalid x-axis size':
                                continue
                            elif result == 'invalid y-axis size':
                                visualizable_tests += 1
                            else:
                                correct_size_tests += 1
                                visualizable_tests += 1

                                # Visualize the result if it is valid
                                data = json.loads(result)
                                visualize.heatmap(f"{model_name}_{learning_rate}_{epoch}", data, round(temperature,2), i)
                
                # Increase temperature to generate more diverse tests
                temperature += 0.1

        # Save model and tokenizer after each epoch to be able to continue training later
        print(f"Saving model...")
        model.save_pretrained(f'{model_name}_{learning_rate}')
        tokenizer.save_pretrained(f'{model_name}_{learning_rate}')
        print(f"Model saved")

        # Visualize the statistics
        visualize.graph(model_name, learning_rate, stats)
        visualize.avg_graph(model_name, learning_rate, stats)

        # Save the statistics for this epoch
        stats.append({
            'epoch': epoch,
            'correct_size_tests': correct_size_tests,
            'visualizable_tests': visualizable_tests
        })
        # Update the statistics file
        os.makedirs(f'stats', exist_ok=True)
        with open(f'stats/{model_name}_{learning_rate}.json', 'w') as file:
            json.dump(stats, file)
            

if __name__ == '__main__':
    # Command line arguments to continue training a model
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--continue_training", help="continue the training of a model")
    args = argParser.parse_args()

    if args.continue_training:
        # Start training GPT-2 model from a previous training
        model_path = args.continue_training
        learning_rate = float(model_path.split('_')[-1])
        batch_size = 1
        parent_model = ''.join(model_path.split('_')[0])
        # Load stats
        with open(f'stats/{parent_model}_{learning_rate}.json', 'r') as file:
            stats = json.load(file)

        train(model_path, learning_rate, batch_size, model_name=parent_model, stats=stats)
    else:
        # Start training GPT-2 model from scratch with default parameters
        model_path = 'gpt2'
        learning_rate = 5e-5
        batch_size = 1
        train(model_path, learning_rate, batch_size)