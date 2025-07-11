import re
from dataclasses import dataclass

import pandas as pd
from math_verify import parse, verify

from .transform import DFTransformBase


@dataclass
class MathVerifyOutputEvaluator(DFTransformBase):
    """
    This class is for evaluating the output of models for the verifiable tasks,
    using Math-Verify library for standardization
    https://github.com/huggingface/Math-Verify
    """

    score_column_name: str = "score"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.score_column_name] = df.apply(
            lambda row: evaluate(row["model_output"], row["ground_truth"]), axis=1
        )
        return df


def evaluate(model_output, answer):
    if not model_output or model_output == "":
        return False

    # extract box
    if "oxed{" not in model_output:
        for flag in [
            "the final answer is",
            "the answer is",
            "the correct answer is",
            "the answer should be",
        ]:
            raw_model_output = model_output
            model_output = model_output.split(flag)[-1].strip()
            if flag in raw_model_output:
                model_output = model_output.split("\n")[0].split(". ")[0]
            flag = flag.replace("the", "The")
            raw_model_output = model_output
            model_output = model_output.split(flag)[-1].strip()
            if flag in raw_model_output:
                model_output = model_output.split("\n")[0].split(". ")[0]
    elif model_output.count("oxed{") > 1:
        model_output = "\\boxed{" + model_output.split("oxed{")[-1]

    if model_output.count("oxed{") >= 1:
        matches = re.findall(r"\\boxed\{(.*?)}", model_output)
        assert matches, f"\\boxed parsing error: {model_output}"
        model_output = matches[-1]

    gt_answer = answer if isinstance(answer, str) else str(answer)

    if gt_answer.lower() not in ["yes", "no", "true", "false"]:
        hypo = parse(model_output)
        gt = parse(gt_answer)
    else:
        hypo = model_output
        gt = gt_answer
    print(model_output.count("oxed{"))
    print(hypo, gt, verify(hypo, gt))
    return verify(hypo, gt)
