# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple
import pdb
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
import scipy
from gsm8k_verifier_metrics import GSM8KVerifierMetrics

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask
import datasets
from deberta_model import DebertaV2ForTokenClassification
import pdb

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    previous_run_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    test_data: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    solution_correct_loss_weight: Optional[float] = field(
        default=1.0, metadata={"help": "help"}
    )
    solution_incorrect_loss_weight: Optional[float] = field(
        default=1.0, metadata={"help": "help"}
    )
    step_correct_loss_weight: Optional[float] = field(
        default=0.1, metadata={"help": "help"}
    )
    step_incorrect_loss_weight: Optional[float] = field(
        default=0.1, metadata={"help": "help"}
    )
    other_label_loss_weight: Optional[float] = field(
        default=1.0, metadata={"help": "help"}
    )
    num_of_multi_task_solution_wise_and_stepwise_both_train_epochs: Optional[int] = field(
        default=5, metadata={"help": "help"}
    )
    num_of_solution_wise_only_train_epochs: Optional[int] = field(
        default=0, metadata={"help": "help"}
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = token_classification_task.get_labels(data_args.data_labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.previous_run_dir is not None:
        model_args.model_name_or_path = model_args.previous_run_dir

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    # pdb.set_trace()

    # code change begin
    config.task_specific_params = {}
    config.task_specific_params["solution_correct_loss_weight"] = data_args.solution_correct_loss_weight
    config.task_specific_params["solution_incorrect_loss_weight"] = data_args.solution_incorrect_loss_weight
    config.task_specific_params["step_correct_loss_weight"] = data_args.step_correct_loss_weight
    config.task_specific_params["step_incorrect_loss_weight"] = data_args.step_incorrect_loss_weight
    config.task_specific_params["other_label_loss_weight"] = data_args.other_label_loss_weight
    # code change end

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = DebertaV2ForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    data_dir = data_args.train_data.replace("train.txt", "")

    # Get datasets
    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    # save the texual sequences of eval dataset
    eval_sequences = [tokenizer.decode(x.input_ids) for x in eval_dataset]

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if j == 1: # only pick the second index
                # if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
        return preds_list, out_label_list

    def get_solution_logits(predictions: np.ndarray):
        scores = []
        for i in range(predictions.shape[0]):
            solution_correct_index = config.label2id["SOLUTION-CORRECT"]
            score = scipy.special.softmax(predictions[i][1])[solution_correct_index].item()
            scores.append(score)
        return scores

    # gsm8k_metric = datasets.load_metric("./gsm8k_verifier_metrics")
    gsm8k_metric = GSM8KVerifierMetrics(eval_sequences=eval_sequences, label2id=config.label2id)

    def compute_metrics(p: EvalPrediction) -> Dict:
        scores = get_solution_logits(p.predictions)
        return gsm8k_metric.compute(predictions=scores, references=scores)
        # preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        # return {
        #     "accuracy_score": accuracy_score(out_label_list, preds_list),
        #     "precision": precision_score(out_label_list, preds_list),
        #     "recall": recall_score(out_label_list, preds_list),
        #     "f1": f1_score(out_label_list, preds_list),
        # }

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

        # # stage 1: Train with both solution-wise label and step-wise label
        # if data_args.num_of_multi_task_solution_wise_and_stepwise_both_train_epochs > 0:
        #     trainer.args.output_dir = os.path.join(trainer.args.output_dir, "both_labels")
        #     trainer.args.num_train_epochs = data_args.num_of_multi_task_solution_wise_and_stepwise_both_train_epochs
        #     trainer.train(
        #         model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        #     )
        #     trainer.save_model()
        #     # For convenience, we also re-save the tokenizer to the same directory,
        #     # so that you can share your model easily on huggingface.co/models =)
        #     if trainer.is_world_process_zero():
        #         tokenizer.save_pretrained(training_args.output_dir)
        
        # # stage 2: Train with solution-wise label only
        # if data_args.num_of_solution_wise_only_train_epochs > 0:
        #     config.task_specific_params["solution_correct_loss_weight"] = 1.0
        #     config.task_specific_params["solution_incorrect_loss_weight"] = 1.0
        #     config.task_specific_params["step_correct_loss_weight"] = 1.0
        #     config.task_specific_params["step_incorrect_loss_weight"] = 1.0
        #     config.task_specific_params["other_label_loss_weight"] = 1.0

        #     if data_args.num_of_multi_task_solution_wise_and_stepwise_both_train_epochs == 0:
        #         trainer.args.output_dir = os.path.join(trainer.args.output_dir, "both_labels")

        #     if len(os.listdir(trainer.args.output_dir)) > 0:
        #         ckpt_num_list = sorted([int(dir.split("-")[-1]) for dir in os.listdir(trainer.args.output_dir) if dir.startswith("checkpoint")], reverse=True)
        #         trainer.args.num_train_epochs = data_args.num_of_solution_wise_only_train_epochs + len(ckpt_num_list)
        #         if len(ckpt_num_list) > 0:
        #             last_ckpt_dir = os.path.join(trainer.args.output_dir, f"checkpoint-{ckpt_num_list[0]}")
        #             continue_model_path = last_ckpt_dir
        #             new_model = DebertaV2ForTokenClassification.from_pretrained(
        #                 continue_model_path,
        #                 from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #                 config=config,
        #                 cache_dir=model_args.cache_dir,
        #             ).to("cuda")
        #             trainer.args.output_dir = os.path.join(trainer.args.output_dir, "..", "solution_wise_label_only")
        #             trainer.model = new_model
        #             trainer.train(model_path=continue_model_path)
        #             trainer.save_model()
                
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, _ = align_predictions(predictions, label_ids)

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(data_dir, "test.txt"), "r") as f:
                    token_classification_task.write_predictions_to_file(writer, f, preds_list)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
