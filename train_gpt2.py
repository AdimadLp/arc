import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
import test_gpt2
import visualize
import argparse

def load_arc_data(directory):
    tasks = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                tasks.append(json.load(file))
    return tasks

def grid_to_string(grid):
    return ' '.join(' '.join(str(cell) for cell in row) for row in grid)

def task_to_string(task):
    pairs = task['train'] + task.get('test', [])
    return '\n'.join(f"Input: {pair['input']} Output: {pair['output']}" for pair in pairs)

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

def train(model_path, learning_rate, batch_size, epoch, model_name=None, stats=[]):
    if model_name is None:
        model_name = model_path

    if stats == []:
        max_correct_size_tests = 0
    else:
        # Check if epoch exists in stats
        epoch_exists_in_stats = any(stat['epoch'] == epoch for stat in stats)
        # Remove all stats after the epoch
        stats = [stat for stat in stats if stat['epoch'] <= epoch]
        if epoch_exists_in_stats:
            # Get the max correct size tests from stats
            max_correct_size_tests = max([stat['correct_size_tests'] for stat in stats])
        else:
            raise Exception("Epoch does not match with the last epoch in stats")
    
    print(f"model_path: {model_path}\nlearning_rate: {learning_rate}\nbatch_size: {batch_size}\nmodel_name: {model_name}\nepoch: {epoch}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # Ensure that pad_token is set
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    def collate_batch(batch):
        input_ids, attention_masks = zip(*batch)
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        return input_ids_padded, attention_masks_padded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print("Model loaded")

    train_data = load_arc_data('training')
    train_dataset = ArcDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    print("Taining data loaded")

    # Load validation data
    val_data = load_arc_data('evaluation')
    val_dataset = ArcDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
    print("Evaluation data loaded")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train model
    print("Start training...")
    while True:  # Number of epochs
        model.train()

        for step, batch in enumerate(train_loader, start=1):
            inputs, masks = batch
            inputs, masks = inputs.to(device), masks.to(device)

            optimizer.zero_grad()  # Reset gradients at the start of each batch
            outputs = model(input_ids=inputs, attention_mask=masks, labels=inputs)
            loss = outputs.loss  # No need to normalize the loss
            loss.backward()

            optimizer.step()  # Update model parameters after each batch

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

        optimizer.step()  # Ensure any remaining gradients are pushed through
        optimizer.zero_grad()

        epoch += 1
        print(f"Epoch {epoch} completed with loss {loss.item()}")

        visualizable_tests = 0
        correct_size_tests = 0
        
        # Evaluate model every epoch by generating 5 tests called from test data '0c786b71.json' and visualize them
        temperature = 0.1

        for l in range(1, 10+1):
                for i in range(1, 3+1):
                            result = test_gpt2.test_model(model, tokenizer, device, temperature)

                            # Check if the result is not empty
                            if result == 'invalid output':
                                continue
                            elif result == 'invalid x-axis size':
                                continue
                            elif result == 'invalid y-axis size':
                                visualizable_tests += 1
                            else:
                                correct_size_tests += 1
                                visualizable_tests += 1
                                # Visualize the result
                                data = json.loads(result)
                                visualize.heatmap(f"{model_name}_{learning_rate}_{epoch}", data, round(temperature,2), i)
                                
                temperature += 0.1
        # first epoch
        if stats == []:
            stats.append({
                'epoch': epoch,
                'correct_size_tests': correct_size_tests,
                'visualizable_tests': visualizable_tests
            })
        # if the number of correct size tests or visualizable tests is higher than the previous epoch, save the model
        elif max_correct_size_tests < correct_size_tests:
            max_correct_size_tests = correct_size_tests
            stats.append({
                'epoch': epoch,
                'correct_size_tests': correct_size_tests,
                'visualizable_tests': visualizable_tests
            })
            # Save model
            print(f"Saving model...")
            model.save_pretrained(f'{model_name}_{learning_rate}_{epoch}')
            tokenizer.save_pretrained(f'{model_name}_{learning_rate}_{epoch}')
            print(f"Model saved")
        else:
            stats.append({
                'epoch': epoch,
                'correct_size_tests': correct_size_tests,
                'visualizable_tests': visualizable_tests
            })

        # Visualize stats
        visualize.graph(model_name, learning_rate, stats)
        visualize.avg_graph(model_name, learning_rate, stats)

        os.makedirs(f'stats', exist_ok=True)
        # Save stats
        with open(f'stats/{model_name}_{learning_rate}.json', 'w') as file:
            json.dump(stats, file)
            

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--continue_training", help="continue the training of a model")

    args = argParser.parse_args()

    if args.continue_training:
        model_path = args.continue_training
        learning_rate = float(model_path.split('_')[-2])
        batch_size = 1
        parent_model = ''.join(model_path.split('_')[:-2])
        epoch = int(model_path.split('_')[-1])
        # Load stats
        with open(f'stats/{parent_model}_{learning_rate}.json', 'r') as file:
            stats = json.load(file)

        train(model_path, learning_rate, batch_size, epoch, model_name=parent_model, stats=stats)
    else:
        model_path = 'gpt2'
        learning_rate = 2e-5
        epoch = 0
        batch_size = 1
        train(model_path, learning_rate, batch_size, epoch)

    
"""
# Evaluate model every epoch
model.eval()
with torch.no_grad():
    total_loss = 0
    for step, batch in enumerate(val_loader, start=1):
        inputs, masks = batch
        inputs, masks = inputs.to(device), masks.to(device)

        outputs = model(input_ids=inputs, attention_mask=masks, labels=inputs)
        loss = outputs.loss
        total_loss += loss.item()
    print(f"Validation loss: {total_loss / step}")"""