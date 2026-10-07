"""
Evaluate baseline OPT-125M on 5 LM-Eval tasks via Python API
"""
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

tasks = ["piqa","hellaswag","arc_easy","arc_challenge","winogrande"]
model = HFLM(pretrained="facebook/opt-125m", device="cpu")
results = evaluator.simple_evaluate(
    model=model,
    tasks=tasks,
    limit=10,
    batch_size=1,
    log_samples=False
)

for task, metrics in results.get('results', {}).items():
    print(task, metrics)
