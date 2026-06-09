import os
from config import EvaluationConfig
from evaluator import Evaluator
from logging_config import logger


yaml_path = os.environ.get(
    "TRUST_EVAL_CONFIG",
    os.path.join(os.path.dirname(__file__), "eval_config.yaml"),
)
evaluation_config = EvaluationConfig.from_yaml(yaml_path=yaml_path)
logger.info(evaluation_config)
evaluator = Evaluator(evaluation_config)
evaluator.compute_metrics()
evaluator.save_results(output_path = "results/asqa_eval_gtr_top100_with_nemo_guidance_refined_0_1_2_trust_eval_result.json")
