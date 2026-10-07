"""
Run lm-eval via Python API to avoid CLI Unicode crash
"""
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model = HFLM(pretrained="facebook/opt-125m", device="cpu")
results = evaluator.simple_evaluate(
    model=model,
    tasks=["piqa"],
    limit=5,
    batch_size=1,
    log_samples=False
)
print("Results keys:", results.keys())
for task, metrics in results.get('results', {}).items():
    print(task, metrics)
