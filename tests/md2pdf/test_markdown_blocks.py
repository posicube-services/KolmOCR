import textwrap


from lab.md2pdf import Block, blocks2html, parse_markdown2blocks


def test_parse_markdown2blocks_smoke():
    markdown = textwrap.dedent(
        """
        # Title
        
        Intro paragraph spanning
        multiple lines.
        
        - First item
        - Second item
        
        ![Alt text](figure.png "Caption text")
        
        | Column A | Column B |
        | -------- | -------- |
        | Foo      | Bar      |
        """
    ).strip()

    blocks = parse_markdown2blocks(markdown)

    assert [block["type"] for block in blocks] == [
        "heading",
        "paragraph",
        "list",
        "figure",
        "table",
    ]

    heading, paragraph, list_block, figure, table = blocks

    assert heading == {"id": "b001", "type": "heading", "content": "Title", "level": 1}
    assert paragraph["content"] == "Intro paragraph spanning multiple lines."
    assert list_block["items"] == ["First item", "Second item"]
    assert list_block["ordered"] is False
    assert figure["id"] == "b004"
    assert figure["type"] == "figure"
    assert figure["src"] == "figure.png"
    assert figure["alt"] == "Alt text"
    assert figure["caption"] == "Caption text"
    assert table["headers"] == ["Column A", "Column B"]
    assert table["rows"] == [["Foo", "Bar"]]


def test_blocks2html_renders_expected_document():
    blocks = [
        Block(id="b001", type="heading", level=2, content="Intro"),
        {"id": "b002", "type": "paragraph", "content": "Text & <stuff>"},
    ]

    html_document = blocks2html(blocks)

    expected = """
<div class="document">
  <div class="block" data-id="b001">
    <h2>Intro</h2>
  </div>
  <div class="block" data-id="b002">
    <p>Text &amp; &lt;stuff&gt;</p>
  </div>
</div>
""".strip()

    assert html_document == expected


def test_parse_markdown_preserves_code_block_lines():
    markdown = textwrap.dedent(
        """
        ```python
        def add(a, b):
            return a + b
        ```
        """
    ).strip()

    blocks = parse_markdown2blocks(markdown)

    assert any("\n" in block["content"] for block in blocks if block["type"] == "paragraph")


def test_parse_markdown_list_item_with_additional_paragraphs():
    markdown = textwrap.dedent(
        """
        - First line
        
            Still the same item
        
        - Second item
        """
    ).strip()

    blocks = parse_markdown2blocks(markdown)
    list_block = next(block for block in blocks if block["type"] == "list")

    assert "Still the same item" in list_block["items"][0]


def test_parse_markdown_skips_front_matter():
    markdown = textwrap.dedent(
        """
        ---
        primary_language: en
        is_rotation_valid: True
        rotation_correction: 0
        ---
        Actual content starts here.
        """
    ).strip()

    blocks = parse_markdown2blocks(markdown)

    assert blocks[0]["type"] == "front_matter"
    first_paragraph = next(block for block in blocks if block["type"] == "paragraph")
    assert first_paragraph["content"].startswith("Actual content")
    assert not any("primary_language" in block.get("content", "") for block in blocks if block["type"] == "paragraph")


def test_blocks2html_includes_front_matter_comment():
    markdown = textwrap.dedent(
        """
        ---
        primary_language: en
        is_rotation_valid: True
        ---
        Body text.
        """
    ).strip()

    blocks = parse_markdown2blocks(markdown)
    html_document = blocks2html(blocks)

    assert html_document.startswith("<!-- front_matter")
    assert "primary_language: en" in html_document
