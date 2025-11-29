from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Node:
    label: str
    children: List["Node"] = field(default_factory=list)

    def add(self, child: "Node") -> None:
        self.children.append(child)


def tree_size(node: Node) -> int:
    return 1 + sum(tree_size(c) for c in node.children)


def tree_edit_distance(
    a: Node,
    b: Node,
    replace_cost_fn: Optional[Callable[[Node, Node], float]] = None,
    size_fn: Optional[Callable[[Node], int]] = None,
) -> float:
    """
    Ordered-tree edit distance with pluggable costs.
    - Substitution cost: replace_cost_fn(a, b) if provided, else 0/1(label match/mismatch)
    - Insertion/deletion cost: size_fn(subtree) if provided, else subtree size
    """
    size = size_fn or tree_size
    if a is None and b is None:
        return 0
    if a is None:
        return size(b)
    if b is None:
        return size(a)

    # Align children with dynamic programming
    a_children = a.children
    b_children = b.children
    m, n = len(a_children), len(b_children)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + size(a_children[i - 1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + size(b_children[j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_sub = tree_edit_distance(a_children[i - 1], b_children[j - 1], replace_cost_fn, size_fn)
            cost_del = size(a_children[i - 1])
            cost_ins = size(b_children[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + cost_del,
                dp[i][j - 1] + cost_ins,
                dp[i - 1][j - 1] + cost_sub,
            )

    replace_cost = replace_cost_fn(a, b) if replace_cost_fn else (0 if a.label == b.label else 1)
    return replace_cost + dp[m][n]
