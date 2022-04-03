import re
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from multiset import Multiset
from functools import lru_cache
import random


class GSM8KCase:
    def __init__(self, ground_truth, preds):
        self.question = ""
        self.ground_truth = ground_truth
        self.preds = preds
        self.correct_preds_num = 0.0


class GSM8KExample:
    inf = "-99999999"
    
    def __init__(self, content):
        self.content = content.strip().replace("\n", "")
        self.steps = self.get_steps()
        self.step_labels = {}
        self.sequence_labels = []
        self.equations = self.init_equations()
        self.is_correct= False
        self.verifier_score = 0.0

    # 按'@@xxx>>'的格式将公式提取出来
    def init_equations(self):
        return [x for x in re.findall("@@.+>>[0-9\.]+", self.content) if "=" in x]

    def get_steps(self):
        return [x+"%%" if x != self.content.split("%%")[-1] else x for i, x in enumerate(self.content.split("%%"))]

    def get_final_answer(self):
        ans = ""
        if "####" in self.content:
            ans = self.content.split("####")[-1].strip().replace("\n", "").replace(" ", "")
        else:
            ans = GSM8KExample.inf
        return clean_ans(ans)

    def get_step_answer(step):
        expression = re.findall("@@.+>>[0-9\.]+", step)
        if len(expression == 0):
            ans = GSM8KExample.inf
        else:
            ans = expression[-1].split(">>")[-1].strip()
        return clean_ans(ans)

    def label_to_string(self):
        return "".join(str(self.labels[k]) for k in self.labels.keys())


def convert_eval_sequences_to_gsm8k_cases(eval_sequences):
    cases = []
    for i in range(0, len(eval_sequences), 101):
        case = GSM8KCase("", [])
        question, grount_truth = eval_sequences[i].split(" && ")[0], eval_sequences[i].split(" && ")[1]
        case.ground_truth = GSM8KExample(grount_truth)
        case.question = question
        for j in range(i+1, i+101):
            case.preds.append(GSM8KExample(eval_sequences[j].split(" && ")[1]))
        cases.append(case)
    return cases


def random_1_hit(gt_ans, preds):
    idx = random.randint(0, len(preds)-1)
    # random 1 acc
    pred0_ans = preds[idx].get_final_answer()
    return 1 if pred0_ans == gt_ans else 0

def recall_hit(gt_ans, preds):
    for pred in preds:
        if pred.get_final_answer() == gt_ans:
            return 1
    return 0

def voting_hit(gt_ans, preds):
    # voting acc
    answers = {}
    for pred in preds:
        if pred.get_final_answer() not in answers:
            answers[pred.get_final_answer()] = 0
        answers[pred.get_final_answer()] += 1
    answers = sorted(answers.items(), key=lambda x : x[1], reverse=True)
    for i in range(len(answers)):
        ans, ans_cnt = answers[i][0], answers[i][1]
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0

def weighted_voting_hit(gt_ans, preds):
    # voting acc
    answers = {}
    for pred in preds:
        if pred.get_final_answer() not in answers:
            answers[pred.get_final_answer()] = 0
        answers[pred.get_final_answer()] += pred.verifier_score
    answers = sorted(answers.items(), key=lambda x : x[1], reverse=True)
    for i in range(len(answers)):
        ans, ans_cnt = answers[i][0], answers[i][1]
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0

def verification_hit(gt_ans, preds):
    preds = sorted(preds, key=lambda x : x.verifier_score, reverse=True)
    for pred in preds:
        ans = pred.get_final_answer()
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0


def compute_results(data, rand_k=100):
    # total_cnt = 0
    total_random_hit_cnt = 0
    total_recall_cnt = 0
    total_vote_cnt = 0
    total_weighted_vote_cnt = 0
    total_verification_cnt = 0
    for i, x in enumerate(data):
        gt_ans = x.ground_truth.get_final_answer()
        slice = x.preds if rand_k == len(x.preds) else random.sample(x.preds, rand_k)
        
        total_random_hit_cnt += random_1_hit(gt_ans, slice)
        total_vote_cnt += voting_hit(gt_ans, slice)
        total_recall_cnt += recall_hit(gt_ans, slice)
        total_weighted_vote_cnt += weighted_voting_hit(gt_ans, slice)
        total_verification_cnt += verification_hit(gt_ans, slice)
        
    return {
        "random_top1": total_random_hit_cnt / len(data), 
        "recall": total_recall_cnt / len(data),
        "verifier_top1_accuracy": total_verification_cnt / len(data),
        "voting_top1_accuracy": total_vote_cnt / len(data),
        "weighted_voting_top1_accuracy": total_weighted_vote_cnt / len(data),
    }


# def compute_metrics_avg(data, rand_k=100, repeat_time=1):
#     sum_result_dict = {
#         "total_random_hit_cnt": 0, 
#         "total_recall_cnt": 0,
#         "total_vote_cnt": 0,
#         "total_weighted_vote_cnt": 0,
#         "total_verification_cnt": 0,
#     }
#     for i in tqdm(range(repeat_time)):
#         for k in sum_result_dict:
#             result_dict = compute_metrics(data, rand_k=rand_k)
#             sum_result_dict[k] += result_dict[k]
#     for k in sum_result_dict:
#         sum_result_dict[k] = sum_result_dict[k] / repeat_time if repeat_time != 1 else sum_result_dict[k]
#     return sum_result_dict


@lru_cache(maxsize=4096)
def get_answer(s):
    ans = ""
    if "####" in s:
        ans = s.split("####")[-1].replace("\n", "").replace(" ", "").strip()
    else:
        expression = re.findall("@@.+>>[0-9\.]+", s)
        if len(expression) == 0:
            ans = GSM8KExample.inf
        else:
            ans = expression[-1].split(">>")[-1].strip()
    return clean_ans(ans)


def match(steps, positive_examples):
    curr_set = Multiset([get_answer(x) for x in steps])
    for positive_example in positive_examples:
        golden_set = Multiset([get_answer(x) for x in positive_example.steps])
        if GSM8KExample.inf in curr_set:
            curr_set.remove(GSM8KExample.inf)
        if GSM8KExample.inf in golden_set:
            golden_set.remove(GSM8KExample.inf)
        if len(curr_set) == 0:
            return 0
        # print(curr_set)
        # print(golden_set)
        if curr_set.issubset(golden_set):
            return 1
    return 0


def get_final_answer(s):
    ans = ""
    if "####" in s:
        ans = s.split("####")[-1].strip().replace("\n", "").replace(" ", "")
    else:
        ans = GSM8KExample.inf
    return clean_ans(ans)
    

def dedup(li):
    s = set()
    new_li = []
    for x in li:
        if str(x) not in s:
            new_li.append(x)
            s.add(str(x))
    return new_li


def print_stat(data):
    cnt = 0
    for x in data:
        if x["output"] == "correct":
            cnt += 1
    print(cnt, len(data) - cnt, len(data))


def clean_ans(s):
    s = str(s)
    if s and len(s) > 0 and s[-1] == '.':
        s = s[:-1]
    return s