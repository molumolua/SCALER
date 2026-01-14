import re
from typing import Dict, Union
def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def extract_answer(sample: str) -> str:
    """Extracts the final answer from the sample."""
    answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', sample, re.DOTALL)
    answer_matched = False
    if answer_matches:
        model_final_output = answer_matches[-1].strip()  # Get the last match
        answer_matched = True
    else:
        # extract the answers between \\boxed{ and } (if there are multiple \\boxed{} pairs, only the last one will be extracted)
        model_final_output = last_boxed_only_string(sample)
        if model_final_output is None:
            model_final_output = sample
        else:
            answer_matched = True

    answer_prefixes = ["The answer is:", "The final answer is ", "The final answer is: ", "The answer is "]
    answer = model_final_output
    for answer_prefix in answer_prefixes:
        answer = answer.replace(answer_prefix, "").strip()
    if answer.endswith("."):
        answer = answer[:-1].strip()
    return strip_latex(answer).lower(), answer_matched



def fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match function for BigBench Extra Hard."""

    prediction = str(prediction).lower()
    reference = str(reference).lower()

    if prediction == reference:
        return True

    # (a) vs a
    if len(str(prediction)) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(str(reference)) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


def extract_answer_content(sample: str) -> tuple[bool, str]:
    prediction, answer_matched = extract_answer(sample.strip())
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    if not answer_matched:
        prediction_in_box = last_boxed_only_string(prediction)
        if prediction_in_box:
            answer_matched = True
            prediction = prediction_in_box

    return prediction, answer_matched


def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def compute_score(predict_str: str, ground_truth: str) -> Dict[str, Union[int, float]] :
    extracted_answer, format_correct = extract_answer_content(predict_str)
    reference = preprocess_reference(ground_truth)
    correctness = fuzzy_match(extracted_answer, reference)
    if correctness:
        acc, reward = 1, 1.0
    else:
        acc, reward = 0, -1.0
    
    return {
        "score": reward,
        "acc":acc,
        "answer":predict_str,
        "pred":extracted_answer,
        "format_verify":0,
    }