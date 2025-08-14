import re
import string
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


LABELS_TRUE = ["yes", "true", "correct"]
LABELS_FALSE = ["no", "false", "incorrect"]
LABELS_MC = list(string.ascii_lowercase[:10])  # MMMU-pro has up to 10 choices


def extract_last_boxed(text):
    pattern = r"\\boxed\{"
    results = []
    for match in re.finditer(pattern, text):
        start = match.end()
        depth = 1
        content = []
        i = start

        while i < len(text):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    results.append("".join(content))
                    break
            content.append(char)
            i += 1

    return results[-1] if results else None


def extract_boxed_answer(reasoning: str):
    """
    Extract the content inside the last \\boxed{...} in the reasoning.
    """
    if reasoning is None:
        return None
    mc_answer = reasoning.strip()
    mc_labels = [*LABELS_TRUE, *LABELS_FALSE]
    if mc_answer.lower() in mc_labels:
        return mc_answer
    elif reasoning.split("\n")[0].lower() in [
        *mc_labels,
        *[f"{v}." for v in mc_labels],
    ]:
        return reasoning.split("\n")[0].removesuffix(".")
    elif reasoning.split("\n")[0].lower() in [
        *[f"\\boxed{{{v}}}" for v in mc_labels],
        *[f"\\boxed{{{v}}}." for v in mc_labels],
    ]:
        out = extract_last_boxed(reasoning.split("\n")[0])
        if out is None:
            return None
        return out.removesuffix(".")
    # matches = re.findall(r"\\boxed\{(.*?)}", reasoning)
    # output = matches[-1] if matches else "Unknown"

    output = extract_last_boxed(reasoning)
    if output is None:
        # openvlthinker
        matches = re.findall(r"<answer>(.*?)</answer>", reasoning)
        output = matches[-1] if matches else None
    if output is None:
        # internvl-4b
        pattern = r"The answer is:\s*(.+?)"
        matches = re.findall(pattern, reasoning, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        pattern = r"The correct answer is:\s*(.+?)"
        matches = re.findall(pattern, reasoning, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is not None:
        # vision-r1: <answer> Final Answer:{}</answer>
        output = (
            output.strip().removeprefix("Final Answer:").removeprefix("Answer:").strip()
        )

    if output is None:
        # vision-r1:
        pattern = r"\*\*Final Answer:\*\*\s*(.+?)\s*</think>$"
        matches = re.findall(
            pattern, reasoning.strip(), flags=re.IGNORECASE | re.DOTALL
        )
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # vision-r1:
        pattern = r"\*\*Final Answer:\*\*\s*(.+?)\s*$"
        matches = re.findall(
            pattern, reasoning.strip(), flags=re.IGNORECASE | re.DOTALL
        )
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # vision-r1:
        pattern = r"\*\*Answer:\*\*\s*(.+?)\s*</think>$"
        matches = re.findall(
            pattern, reasoning.strip(), flags=re.IGNORECASE | re.DOTALL
        )
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # vision-r1:
        pattern = r"\*\*Answer:\*\*\s*(.+?)\s*$"
        matches = re.findall(
            pattern, reasoning.strip(), flags=re.IGNORECASE | re.DOTALL
        )
        output = matches[-1].removeprefix(".") if matches else None

    if output is None:
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        pattern = r"<CONCLUSION>(.+?)</CONCLUSION>"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        temp = matches[-1].removeprefix(".") if matches else None
        if temp is not None:
            pattern = r"is\s+([^.]+)\."

            # find all matches, take last one
            matches = re.findall(pattern, temp, flags=re.IGNORECASE)
            output = matches[-1].strip() if matches else None
            if output is None:
                output = temp
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        pattern = r"Therefore, the final answer is\s+\\\(\s*(.+?)\s*\\\)"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        pattern = r"Therefore, the answer is\s+\\\(\s*(.+?)\s*\\\)"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        pattern = r"Thus, the final answer is\s+\\\(\s*(.+?)\s*\\\)"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        pattern = r"Thus, the answer is\s+\\\(\s*(.+?)\s*\\\)"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        pattern = r"<\|begin_of_box\|>(.+?)<\|end_of_box\|>"
        matches = re.findall(pattern, reasoning, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1]
        pattern = r"Answer: (.+?)$"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        if last.startswith("Therefore,"):
            pattern = r"is (.+?)$"
            matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
            output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        pattern = r"Therefore, the final answer is\s+\s*(.+?)\s*$"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        pattern = r"Therefore, the answer is\s+\s*(.+?)\s*$"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is None:
        # glm-v4.1
        last = reasoning.strip().split("\n")[-1].removesuffix(".")
        pattern = r"\*\*Final Answer:\*\*\s*(.+?)\s*$"
        matches = re.findall(pattern, last, flags=re.IGNORECASE | re.DOTALL)
        output = matches[-1].removeprefix(".") if matches else None
    if output is not None and "=" in output:
        output = output.split("=")[-1].strip()
    if output is not None:
        output = output.removesuffix("degrees").strip()
    return output


# def extract_boxed_answer(reasoning: str):
#     """
#     Extract the content inside the last \\boxed{...} in the reasoning.
#     """
#     if reasoning is None:
#         return None
#     mc_answer = reasoning.strip()
#     mc_labels = [*LABELS_TRUE, *LABELS_FALSE]
#     if mc_answer.lower() in mc_labels:
#         return mc_answer
#     elif reasoning.split("\n")[0].lower() in [
#         *mc_labels,
#         *[f"{v}." for v in mc_labels],
#     ]:
#         return reasoning.split("\n")[0].removesuffix(".")
#     elif reasoning.split("\n")[0].lower() in [
#         *[f"\\boxed{{{v}}}" for v in mc_labels],
#         *[f"\\boxed{{{v}}}." for v in mc_labels],
#     ]:
#         out = extract_last_boxed(reasoning.split("\n")[0])
#         if out is None:
#             return None
#         return out.removesuffix(".")
#     # matches = re.findall(r"\\boxed\{(.*?)}", reasoning)
#     # output = matches[-1] if matches else "Unknown"
#     output = extract_last_boxed(reasoning)
#     if output is None:
#         # openvlthinker
#         matches = re.findall(r"<answer>(.*?)</answer>", reasoning)
#         output = matches[-1] if matches else None
#     if output is None:
#         # internvl-4b
#         pattern = r"The answer is:\s*(.+?)"
#         matches = re.findall(pattern, reasoning, flags=re.IGNORECASE | re.DOTALL)
#         output = matches[-1].removeprefix(".") if matches else None
#     if output is None:
#         pattern = r"The correct answer is:\s*(.+?)"
#         matches = re.findall(pattern, reasoning, flags=re.IGNORECASE | re.DOTALL)
#         output = matches[-1].removeprefix(".") if matches else None
#     if output is not None:
#         # vision-r1: <answer> Final Answer:{}</answer>
#         output = (
#             output.strip().removeprefix("Final Answer:").removeprefix("Answer:").strip()
#         )
#     return output


def _parse(x):
    parsed = parse(x)
    if len(parsed) != 2:
        hypo = x
    else:
        hypo = parsed[1]
    return hypo


def verify_single(hypo, gt):
    hypo = hypo.strip()
    if gt.lower() in [*LABELS_TRUE, *LABELS_FALSE]:
        is_valid = hypo.lower() in [*LABELS_TRUE, *LABELS_FALSE]

        hypo_true = hypo.lower() in LABELS_TRUE
        gt_true = gt.lower() in LABELS_TRUE
        return hypo_true == gt_true, is_valid
    elif gt.lower() in [*LABELS_MC]:
        is_valid = hypo.lower() in [*LABELS_MC]

        return hypo.lower() == gt.lower(), is_valid
    else:
        hypo = _parse(hypo)
        flag = hypo and len(hypo) < 50
        gt = _parse(gt)
        if flag:
            return verify(hypo, gt), True
        else:
            return False, False


def _do_verify(hypo, gt) -> tuple[bool, bool]:
    hypo = extract_boxed_answer(hypo)
    if hypo is None:
        return False, False

    matches = re.findall(r"\\text\{(.*?)}", hypo)
    if matches:
        hypo = matches[-1]

    if isinstance(gt, str):
        gt = [gt]

    correct = False
    valid = False
    for _gt in gt:
        _correct, _valid = verify_single(hypo, _gt)
        correct = correct or _correct
        valid = valid or _valid
    return correct, valid


def evaluate(hypo, gt):
    if not hypo or hypo == "":
        return False

    return _do_verify(hypo, gt)[0]
