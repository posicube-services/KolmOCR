import re
from typing import List

from olmocr.kolmocr_eval.utils.tree import Node

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)")
CODE_FENCE_PATTERN = re.compile(r"^\s*```")
CODE_FENCE_LANG_PATTERN = re.compile(r"^\s*```([A-Za-z0-9_+\-]+)?")
TABLE_START_PATTERN = re.compile(r"<table", re.IGNORECASE)
TABLE_END_PATTERN = re.compile(r"</table>", re.IGNORECASE)
IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")
LIST_PATTERN = re.compile(r"^[ \t]*[-*+]\s+.+")


def _add_to_parent(stack: List[Node], node: Node, level: int = None):
    # If level is provided (heading), trim stack to that depth then push
    if level is not None:
        while len(stack) > level:
            stack.pop()
        stack[-1].add(node)
        stack.append(node)
    else:
        stack[-1].add(node)


def build_structure_tree(md: str) -> Node:
    """
    Build a simple structural tree from markdown.
    Headings form the hierarchy; other block types attach to the current heading.
    """
    root = Node("doc")
    stack = [root]  # stack depth = current heading depth (root depth 0)
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Heading
        m = HEADING_PATTERN.match(line)
        if m:
            level = len(m.group(1))
            node = Node(f"heading{level}")
            _add_to_parent(stack, node, level)
            i += 1
            continue

        # Code block (fenced)
        if CODE_FENCE_PATTERN.match(line):
            i += 1
            while i < len(lines) and not CODE_FENCE_PATTERN.match(lines[i]):
                i += 1
            # consume closing fence if present
            if i < len(lines):
                i += 1
            _add_to_parent(stack, Node("code"))
            continue

        # Table block
        if TABLE_START_PATTERN.search(line):
            i += 1
            while i < len(lines) and not TABLE_END_PATTERN.search(lines[i]):
                i += 1
            if i < len(lines):
                i += 1
            _add_to_parent(stack, Node("table"))
            continue

        # Image
        if IMAGE_PATTERN.search(line):
            _add_to_parent(stack, Node("image"))
            i += 1
            continue

        # List item
        if LIST_PATTERN.match(line):
            _add_to_parent(stack, Node("list_item"))
            i += 1
            continue

        # Paragraph/other content
        if line.strip():
            _add_to_parent(stack, Node("paragraph"))
        i += 1

    return root


def build_code_tree(md: str) -> Node:
    """
    코드 블록만 추출하여 트리로 구성.
    - 루트(doc) 아래에 code:<lang> 노드를 추가하고,
    - 블록 내부 코드를 원문 라인 기반 트리로 변환해 자식으로 붙인다.
    지원 언어: python, c, c++, cpp, java (대소문자 무시). 언어 미표기/기타 언어도 code:unknown으로 추가한다.
    """
    allowed = {"python", "py", "c", "c++", "cpp", "java"}
    root = Node("doc")
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = CODE_FENCE_LANG_PATTERN.match(line)
        if m:
            lang = (m.group(1) or "").lower()
            body = []
            # consume until closing fence
            i += 1
            while i < len(lines) and not CODE_FENCE_PATTERN.match(lines[i]):
                body.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # consume closing fence

            label = f"code:{lang}" if lang and lang in allowed else "code:unknown"
            code_root = Node(label)
            # 블록 내부를 트리로 변환 (원문 라인 사용)
            for child in _build_code_block_tree(lang, body):
                code_root.add(child)
            root.add(code_root)
            continue
        i += 1
    return root


def _build_code_block_tree(lang: str, body_lines: list[str]) -> list[Node]:
    """
    코드 블록 내부를 간단 트리로 변환.
    - python: 들여쓰기 기반 계층
    - c/c++/cpp/java: { } 블록 기반 계층
    - 기타/미표기: 평면 line 노드
    라인 내용은 원문 그대로 사용한다.
    """
    lang = lang or ""
    raw_lines = [l for l in body_lines]  # keep blanks and spaces
    if not raw_lines:
        return []

    if lang in {"python", "py"}:
        return _build_indent_tree(raw_lines)
    if lang in {"c", "c++", "cpp", "java"}:
        return _build_brace_tree(raw_lines)
    return [Node(f"line:{ln}") for ln in raw_lines]


def _build_indent_tree(lines: list[str]) -> list[Node]:
    root = Node("block")
    stack = [(0, root)]  # (indent, node)
    for ln in lines:
        indent = len(ln) - len(ln.lstrip(" \t"))
        content = ln
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        node = Node(f"line:{content}")
        parent.add(node)
        stack.append((indent + 1, node))
    return root.children


def _build_brace_tree(lines: list[str]) -> list[Node]:
    root = Node("block")
    stack = [root]
    for ln in lines:
        open_count = ln.count("{")
        close_count = ln.count("}")
        content = ln.rstrip("\n")
        if content.strip("{} \t") != "":
            stack[-1].add(Node(f"line:{content}"))
        # open braces create new block
        for _ in range(open_count):
            new_block = Node("block")
            stack[-1].add(new_block)
            stack.append(new_block)
        # close braces pop
        for _ in range(close_count):
            if len(stack) > 1:
                stack.pop()
    return root.children
def _normalize_line(line: str) -> str:
    return " ".join(line.strip().split())
