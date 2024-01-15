from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
import torch

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def test_model(model, tokenizer, device, temperature):
    model.eval()
    with torch.no_grad():
        with open(os.path.join('test', '0c786b71.json'), 'r') as file:
            test_file = json.load(file)
        
        def test_task_to_string(task):
            pairs = task['train'] + task.get('test', [])
            return '\n'.join(f"Input: {pair['input']} Output: {pair.get('output', '[[')}" for pair in pairs)

        test_str = test_task_to_string(test_file)
        tokenized_data = tokenizer(test_str, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        
        outputs = model.generate(
            tokenized_data['input_ids'].to(device),
            do_sample=True,
            max_length=1024,
            pad_token_id=model.config.eos_token_id,
            temperature=0.7,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        def evaluate(result):
            # Split the result string into separate data points
            data_points = result.split("\n")

            # Initialize an empty list to store the data
            data = []

            # size of the first output array
            x_axis_size = 0
            y_axis_size = 0
            # Iterate over the data points, exccept the last one which is not a valid data point
            for i, data_point in enumerate(data_points):
                # Split the data point into input and output parts
                parts = data_point.split(" Output: ")
                input_part = parts[0].replace("Input: ", "")
                output_part = parts[1]

                # Convert the input and output parts from string to list
                input_data = json.loads(input_part)
                output_data = json.loads(output_part)

                # set the size of the first output array
                if i == 0:
                    x_axis_size = len(output_data[0])
                    y_axis_size = len(output_data)
                else:
                    # check if x-axis is the same size
                    if not all(len(x) == x_axis_size for x in output_data):
                        return 'invalid x-axis size'
                    # check if y-axis is the same size
                    if not len(output_data) == y_axis_size:
                        return 'invalid y-axis size'
                    
                # Create a dictionary with the input and output data
                data_dict = {
                    "input": input_data,
                    "output": output_data
                }

                # Add the dictionary to the list
                data.append(data_dict)

            # Convert the list to a JSON-formatted string
            json_data = json.dumps(data)

            return json_data
        
        try:
            return evaluate(generated)
        except json.decoder.JSONDecodeError:
            return 'invalid output'
        except ValueError as e:
            return 'invalid output'
        except IndexError as e:
            return 'invalid output'
        
if __name__ == '__main__':
    path = 'gpt2_2e-05_240'
    device = torch.device("cpu")
    model = load_model(path).to(device)
    tokenizer = load_tokenizer(path)
    test_model(model, tokenizer, device)