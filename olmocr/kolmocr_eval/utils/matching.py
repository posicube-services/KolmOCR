import numpy as np
from Levenshtein import distance as edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

INF = 100000000000


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute BLEU similarity between two texts."""
    smooth = SmoothingFunction().method1
    bleu_score = sentence_bleu([text1.split()], text2.split(), smoothing_function=smooth)
    return bleu_score


def match_texts(pred: list, target: list, threshold: int) -> tuple[dict, dict]:
    def compute_cost_matrix(p_list, t_list):
        return np.array([[edit_distance(p, t) for t in t_list] for p in p_list])

    p2t_matching_dict = {}
    cost_matrix = compute_cost_matrix(pred, target)
    for i in range(len(pred)):
        min_idx = int(np.argmin(cost_matrix[i]))
        if cost_matrix[i][min_idx] <= threshold:
            p2t_matching_dict[i] = min_idx
            cost_matrix[:, min_idx].fill(INF)

    t2p_matching_dict = {}
    cost_matrix = compute_cost_matrix(target, pred)
    for i in range(len(target)):
        min_idx = int(np.argmin(cost_matrix[i]))
        if cost_matrix[i][min_idx] <= threshold:
            t2p_matching_dict[i] = min_idx
            cost_matrix[:, min_idx].fill(INF)

    return p2t_matching_dict, t2p_matching_dict


def f1_score(precision: float, recall: float) -> float:
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
