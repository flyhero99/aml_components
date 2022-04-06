import json
import random
from transformers import AutoTokenizer
import argparse
import tqdm
import pdb
import re
from gsm8k_utils import (
    GSM8KCase,
    GSM8KExample,
    convert_eval_sequences_to_gsm8k_cases,
    compute_results,
)


def get_sequence_labels(question, pred):
    sequence_labels = []
    if pred.is_correct:
        sequence_labels.append(("[CLS]", "SOLUTION-CORRECT"))
    else:
        sequence_labels.append(("[CLS]", "SOLUTION-INCORRECT"))

    # add step tokens
    for s in pred.steps:
        token_list = [x for x in re.split("(>>| )", s) if x != ' ']
        for token in token_list:
            if token == ">>":
                if pred.step_labels[s] == 1:
                    sequence_labels.append((token, "STEP-CORRECT"))
                else:
                    sequence_labels.append((token, "STEP-INCORRECT"))
            else:
                sequence_labels.append((token, "O"))

    # add a split symbol
    sequence_labels.append(("&&", "O"))

    # add question tokens
    for token in question.split(" "):
        sequence_labels.append((token, "O"))

    return sequence_labels


if __name__ == '__main__':
    print("Hello world!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_result_file", type=str, default=None, help="generator output file in .jsonl format")
    parser.add_argument("--diverse_prompt_data_file", type=str, default=None, help="diverse_prompt_data_file")
    parser.add_argument("--output_file", type=str, default=None, help="output file in .jsonl format")
    parser.add_argument("--random_seed", type=int, default=233, help="random_seed")
    parser.add_argument("--split", type=str, default=train, help="split (train or test)")
    args = parser.parse_args()

    random.seed(args.random_seed)

    # loading data from file
    generator_outputs = [json.loads(line) for line in open(args.generator_result_file)]
    diverse_prompt_data = [json.loads(line) for line in open(args.diverse_prompt_data_file)]
    generator_outputs_dict = {x["context"]: x for x in generator_outputs}
    diverse_prompt_data_dict = {x["context"]: x for x in diverse_prompt_data}

    prompt_data = []
    for generator_output in generator_outputs:
        context = generator_output["context"]
        sample = generator_output["samples"][0]
        metadata = diverse_prompt_data_dict[context]["metadata"]
        prompt_data.append({"context": context, "sample": sample, "metadata": metadata})

    prompt_data_dict = {}

    # some pre-processing about formulas and answers
    for obj in tqdm(prompt_data):
        question = obj["context"]
        sample = obj["sample"]
        if "####" not in sample:
            reg = "@@.+>>[\d\.]+"
            eqs = re.findall(reg, sample)
            if len(eqs) > 0:
                final_answer = eqs[-1].split(">>")[-1].strip()
                if final_answer and len(final_answer) > 0 and final_answer[-1] == '.':
                    final_answer = final_answer[:-1]
                # print(final_answer)
                if sample[-2:] == "%%":
                    sample = sample + "####" + final_answer
                else:
                    sample = sample + "%%####" + final_answer
            # print(final_answer)
        if question not in prompt_data_dict:
            prompt_data_dict[question] = []
        prompt_data_dict[question].append(sample)

    # converting data into GSM8KCase
    prompt_cases = []
    prompt_cases_dict = {}
    for k in prompt_data_dict:
        case = GSM8KCase("", [])
        case.question = k
        case.ground_truth = GSM8KExample(diverse_prompt_data_dict[k]["metadata"]["ground_truth"])
        for x in prompt_data_dict[k]:
            pred = Example(x)
            case.preds.append(pred)
        prompt_cases.append(case)
    print("Total cases:", len(prompt_cases))
    print("Case 0's question:", prompt_cases[0].question)
    print("Case 0's ground truth:", prompt_cases[0].ground_truth.content)
    print("Case 0's sample0:", prompt_cases[0].preds[0].content)

    # Step-wise Labeling
    for j, case in enumerate(tqdm(prompt_cases)):
        positive_preds = []
        # 将ground_truth标记为true
        case.ground_truth.is_correct = True
        for step in case.ground_truth.steps:
            case.ground_truth.step_labels[step] = 1
        # 先预存正样本集合
        for i, pred in enumerate(case.preds):
            if pred.get_final_answer() != inf and pred.get_final_answer() == case.ground_truth.get_final_answer():
                positive_preds.append(pred)
        # 再对所有样本的所有step打标签
        for i, pred in enumerate(case.preds):
            if pred.get_final_answer() != inf and pred.get_final_answer() == case.ground_truth.get_final_answer():
                pred.is_correct = True
                for step in pred.steps:
                    pred.step_labels[step] = 1
            else:
                for k, step in enumerate(pred.steps):
                    ans = match(pred.steps[:k+1], positive_preds)
                    pred.step_labels[step] = ans
    
    for case_idx, case in enumerate(tqdm(prompt_cases)):
        case.ground_truth.sequence_labels = get_sequence_labels(case.question, case.ground_truth)
        for pred_idx, pred in enumerate(case.preds):
            pred.sequence_labels = get_sequence_labels(case.question, pred)
    
    sequence_data = []
    for case_idx, case in enumerate(tqdm(prompt_cases)):
        sequence_data.append(case.ground_truth.sequence_labels)
        for pred_idx, pred in enumerate(case.preds):
            sequence_data.append(pred.sequence_labels)

    # Train file is shuffled, but test file is not
    if args.split == "train":
        random.shuffle(sequence_data)
    
    with open(args.output_file, "w") as f:
        for i, arr in enumerate(tqdm(sequence_data)):
            for lhs, rhs in arr:
                f.write(f"{lhs} {rhs}\n")
            f.write("\n")