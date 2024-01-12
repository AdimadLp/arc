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

def test_model(model, tokenizer, device):
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
            top_k=50,
            top_p=0.95,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        def convert_to_json(result):
            # Split the result string into separate data points
            data_points = result.split("\n")

            # Initialize an empty list to store the data
            data = []

            # Iterate over the data points, exccept the last one which is not a valid data point
            for data_point in data_points:
                # Split the data point into input and output parts
                parts = data_point.split(" Output: ")
                input_part = parts[0].replace("Input: ", "")
                output_part = parts[1]

                # Convert the input and output parts from string to list
                input_data = json.loads(input_part)
                output_data = json.loads(output_part)

                # check if all inputs and all outputs have the same length
                if not all(len(x) == len(input_data[0]) for x in input_data):
                    # throw exception if not
                    raise ValueError(f"Input data has different lengths: {input_data}")
                
                if not all(len(x) == len(output_data[0]) for x in output_data):
                    raise ValueError(f"Output data has different lengths: {output_data}")

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
            return convert_to_json(generated)
        except json.JSONDecodeError:
            print(f"Generated text is not JSON compatible: {generated}")
        except ValueError as e:
            print(e)
        
if __name__ == '__main__':
    path = 'arc_model_2e-05_40'
    device = torch.device("cpu")
    model = load_model(path).to(device)
    tokenizer = load_tokenizer(path)
    test_model(model, tokenizer, device)