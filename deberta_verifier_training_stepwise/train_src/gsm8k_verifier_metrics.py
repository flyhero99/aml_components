import absl  # Here to have a nice missing dependency error message early on
import nltk  # Here to have a nice missing dependency error message early on
import numpy  # Here to have a nice missing dependency error message early on
import six  # Here to have a nice missing dependency error message early on
from rouge_score import rouge_scorer, scoring
import datasets
import pdb
import numpy as np
import scipy

from gsm8k_utils import (
    GSM8KCase,
    GSM8KExample,
    convert_eval_sequences_to_gsm8k_cases,
    compute_results,
)

_CITATION = ""
_DESCRIPTION = ""
_KWARGS_DESCRIPTION = ""


def simple_accuracy(preds, labels):
    correct_case_num = 0
    for pred, label in zip(preds, labels):
        pred = pred.replace(" ", "")
        label = label.replace(" ", "")
        if pred == label:
            correct_case_num += 1
    return correct_case_num / len(preds)


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class GSM8KVerifierMetrics(datasets.Metric):
    def __init__(self, eval_sequences=None, label2id=None, **kwargs,):
        super().__init__(**kwargs)
        self.label2id = label2id
        self.cases = convert_eval_sequences_to_gsm8k_cases(eval_sequences)
        # pdb.set_trace()
    
    def assign_scores(self, predictions):
        # pdb.set_trace()
        for i in range(0, len(predictions), 101):
            self.cases[i//101].ground_truth.verifier_score = predictions[i]
            # print("i//101:", i//101)
            for j in range(0, 100):
                # print("i//101:", i//101, "j+1:", j+1, "i+j+1:", i+j+1)
                self.cases[i//101].preds[j].verifier_score = predictions[i+j+1]
    
    def _compute(self, predictions=None, references=None):
        # pdb.set_trace()
        self.assign_scores(predictions)
        return compute_results(self.cases)
    
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32", id="scores"),
                    "references": datasets.Value("float32", id="scores"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )
    
    def _metric_info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )


# metric = GSM8KVerifierMetrics(eval_sequences=[], label_map={})