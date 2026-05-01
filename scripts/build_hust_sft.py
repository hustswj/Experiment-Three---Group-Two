#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import zlib
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


KEYWORDS = [
    '研究生', '博士生', '硕士生', '培养', '培养方案', '培养计划', '培养目标',
    '课程', '学分', '资格考试', '开题', '中期考核', '学业年度进展报告',
    '学术活动', '实践', '专业实践', '学位', '学位论文', '论文', '答辩',
    '毕业', '提前毕业', '学籍', '请假', '注册', '奖学金', '学术道德'
]


QUESTION_TEMPLATES = [
    '根据《2025研究生手册》，{topic}的主要规定是什么？',
    '请依据《2025研究生手册》说明：{topic}。',
    '《2025研究生手册》中关于{topic}有哪些要求？',
]


def read_pdf_objects(pdf_path):
    data = Path(pdf_path).read_bytes()
    objects = {}
    for match in re.finditer(rb'(\d+)\s+0\s+obj\s*(.*?)\s*endobj', data, re.S):
        objects[int(match.group(1))] = match.group(2)
    return objects


def decode_stream(obj_data):
    match = re.search(rb'stream\r?\n(.*?)\r?\nendstream', obj_data, re.S)
    if not match:
        return None
    raw = match.group(1)
    if b'/FlateDecode' in obj_data:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return None
    return raw


def parse_cmap(cmap_bytes):
    text = cmap_bytes.decode('latin1', errors='replace')
    mapping = {}

    for section in re.finditer(r'beginbfchar(.*?)endbfchar', text, re.S):
        for src, dst in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', section.group(1)):
            mapping[int(src, 16)] = bytes.fromhex(dst).decode('utf-16-be', errors='ignore')

    for section in re.finditer(r'beginbfrange(.*?)endbfrange', text, re.S):
        body = section.group(1)
        for start, end, dst in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', body):
            start_i, end_i, dst_i = int(start, 16), int(end, 16), int(dst, 16)
            for code in range(start_i, end_i + 1):
                mapping[code] = chr(dst_i + code - start_i)
        for start, _end, array in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]', body, re.S):
            start_i = int(start, 16)
            for offset, dst in enumerate(re.findall(r'<([0-9A-Fa-f]+)>', array)):
                mapping[start_i + offset] = bytes.fromhex(dst).decode('utf-16-be', errors='ignore')

    return mapping


def build_font_maps(objects):
    font_maps = {}
    for obj_id, obj_data in objects.items():
        if b'/ToUnicode' not in obj_data:
            continue
        match = re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', obj_data)
        if not match:
            continue
        cmap_obj_id = int(match.group(1))
        cmap_stream = decode_stream(objects.get(cmap_obj_id, b''))
        if cmap_stream:
            font_maps[obj_id] = parse_cmap(cmap_stream)
    return font_maps


def decode_hex_text(hex_text, cmap):
    hex_text = ''.join(hex_text.split())
    chars = []
    for idx in range(0, len(hex_text) - 3, 4):
        code = int(hex_text[idx:idx + 4], 16)
        chars.append(cmap.get(code, ''))
    return ''.join(chars)


def extract_text_blocks(objects, font_maps):
    blocks = []
    token_re = re.compile(r'/(FT\d+)\s+[\d.]+\s+Tf|<([0-9A-Fa-f\s]+)>\s*Tj|\[(.*?)\]\s*TJ|\bET\b', re.S)
    array_hex_re = re.compile(r'<([0-9A-Fa-f\s]+)>')

    for obj_id in sorted(objects):
        stream = decode_stream(objects[obj_id])
        if not stream or b'BT' not in stream:
            continue
        content = stream.decode('latin1', errors='replace')
        current_font = None
        parts = []

        for match in token_re.finditer(content):
            font_name, hex_text, array_text = match.group(1), match.group(2), match.group(3)
            if font_name:
                current_font = int(font_name[2:])
                continue

            if hex_text and current_font in font_maps:
                parts.append(decode_hex_text(hex_text, font_maps[current_font]))
                continue

            if array_text and current_font in font_maps:
                for item in array_hex_re.findall(array_text):
                    parts.append(decode_hex_text(item, font_maps[current_font]))
                continue

            if match.group(0) == 'ET' and parts and not parts[-1].endswith('\n'):
                parts.append('\n')

        block = ''.join(parts).strip()
        if block:
            blocks.append(block)
    return blocks


def normalize_lines(text):
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r'\s+', '', raw_line.strip())
        if not line:
            continue
        if re.fullmatch(r'[IVXLCDM]+', line):
            continue
        if re.fullmatch(r'[.\-—·…]+', line):
            continue
        lines.append(line)

    merged = []
    buffer = ''
    heading_re = re.compile(r'^(一、|二、|三、|第[一二三四五六七八九十百零〇]+[章节条]|华中科技大学.+(规定|办法|细则|通知|守则))')

    for line in lines:
        starts_heading = bool(heading_re.match(line))
        if starts_heading and buffer:
            merged.append(buffer)
            buffer = ''

        if not buffer:
            buffer = line
        else:
            buffer += line

        if re.search(r'[。；？！]$', line) or starts_heading:
            merged.append(buffer)
            buffer = ''

    if buffer:
        merged.append(buffer)

    return merged


def extract_articles(lines):
    text = '\n'.join(lines)
    pattern = re.compile(r'(?m)^第[一二三四五六七八九十百零〇]+条')
    starts = [match.start() for match in pattern.finditer(text)]
    articles = []
    seen = set()
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        article = re.sub(r'\n+', '', text[start:end]).strip()
        article = re.sub(r'\s+', '', article)
        if len(article) > 1200:
            article = article[:1200]
        if len(article) < 35 or article in seen:
            continue
        seen.add(article)
        articles.append(article)
    return articles


def topic_from_article(article):
    title = re.split(r'[。；：，,]', article, maxsplit=1)[0]
    title = re.sub(r'^第[一二三四五六七八九十百零〇]+条', '', title).strip()
    if not title:
        title = '研究生培养相关事项'
    if len(title) > 28:
        title = title[:28]
    return title


def build_examples(articles, max_examples):
    selected = []
    for article in articles:
        if any(keyword in article for keyword in KEYWORDS):
            selected.append(article)

    examples = []
    seen_outputs = set()
    for idx, article in enumerate(selected):
        output = f'根据手册原文：{article}'
        if output in seen_outputs:
            continue
        seen_outputs.add(output)
        topic = topic_from_article(article)
        prompt = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)].format(topic=topic)
        examples.append({
            'conversations': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': output}
            ]
        })
        if len(examples) >= max_examples:
            break
    return examples


def main():
    parser = argparse.ArgumentParser(description='Extract HUST graduate handbook text and build SFT jsonl.')
    parser.add_argument('--pdf', default='2025研究生手册.pdf', help='手册 PDF 路径')
    parser.add_argument('--text-output', default='outputs/hust_graduate_handbook_extracted.txt', help='抽取文本输出路径')
    parser.add_argument('--jsonl-output', default='dataset/hust_graduate_sft.jsonl', help='SFT JSONL 输出路径')
    parser.add_argument('--max-examples', type=int, default=80, help='最多生成样本数')
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = ROOT_DIR / pdf_path
    if not pdf_path.exists():
        print(f'未找到 PDF：{pdf_path}', file=sys.stderr)
        sys.exit(1)

    objects = read_pdf_objects(pdf_path)
    font_maps = build_font_maps(objects)
    blocks = extract_text_blocks(objects, font_maps)
    if not blocks:
        print('未能从 PDF 中抽取到文本。', file=sys.stderr)
        sys.exit(1)

    normalized_lines = normalize_lines('\n'.join(blocks))
    articles = extract_articles(normalized_lines)
    examples = build_examples(articles, args.max_examples)
    if not examples:
        print('未能从手册文本中构造培养相关 SFT 样本。', file=sys.stderr)
        sys.exit(1)

    text_output = ROOT_DIR / args.text_output
    jsonl_output = ROOT_DIR / args.jsonl_output
    text_output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)

    text_output.write_text('\n'.join(normalized_lines) + '\n', encoding='utf-8')
    with jsonl_output.open('w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f'抽取文本行数：{len(normalized_lines)}')
    print(f'识别条款数：{len(articles)}')
    print(f'生成 SFT 样本数：{len(examples)}')
    print(f'文本输出：{text_output.relative_to(ROOT_DIR)}')
    print(f'JSONL 输出：{jsonl_output.relative_to(ROOT_DIR)}')


if __name__ == '__main__':
    main()
