import json
import random
from transformers import AutoTokenizer
import argparse
import tqdm
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None, help="input file in .jsonl format")
    parser.add_argument("--output_file", type=str, default=None, help="output file in .jsonl format")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="gpt2", help="tokenizer name, default is \"gpt2\"")
    parser.add_argument("--num_of_examples_in_one_prompt", type=int, default=8, help="num_of_examples_in_one_prompt")
    parser.add_argument("--num_of_prompts_for_one_case", type=int, default=20, help="num_of_prompts_for_one_case")
    parser.add_argument("--num_of_total_examples", type=int, default=-1, help="num_of_total_examples")
    parser.add_argument("--random_seed", type=int, default=233, help="random_seed")
    parser.add_argument("--tokenizer_max_length", type=int, default=1800, help="tokenizer_max_length")
    parser.add_argument("--raw_json_data_input_key", type=str, default="question", help="The keyword of input (like \"input\"), default is \"question\" as for GSM8K.")
    parser.add_argument("--raw_json_data_output_key", type=str, default="answer", help="The keyword of output (like \"output\"), default is \"answer\" as for GSM8K.")
    args = parser.parse_args()

    # original data loading
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        data.append(json.loads(line))
    random.seed(args.random_seed)
    random.shuffle(data)
    # data slicing
    if args.num_of_total_examples != -1:
        data = data[:args.num_of_total_examples]

    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    res = []
    for example in tqdm.tqdm(data):
        for _ in range(args.num_of_prompts_for_one_case):
            prompts = []
            while len(prompts) == 0 or example in prompts:
                prompts = random.sample(data, args.num_of_examples_in_one_prompt)
            p = '\n\n'.join([('Question:\n' + prompt['question'] + '\nAnswer:' + prompt['answer']) for prompt in prompts]) + '\n\nQuestion:\n' + example['question'] + '\nAnswer:'
            if len(tokenizer(p)['input_ids']) < args.tokenizer_max_length:
                res += json.dumps({
                    'context': p,
                    'completion': example['answer'] + '\n\n',
                    'metadata': {
                        'split': args.split,
                        'question': example[args.raw_json_data_input_key],
                        'ground_truth': example[args.raw_json_data_output_key]
                    },
                }) + '\n'
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.writelines(res)