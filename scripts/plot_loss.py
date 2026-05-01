#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections import defaultdict


os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-cache')
REQUIRED_COLUMNS = {'stage', 'epoch', 'step', 'loss', 'learning_rate', 'timestamp'}


def load_loss_rows(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'未找到 loss 日志：{input_path}')

    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f'loss 日志缺少字段：{", ".join(sorted(missing))}')

        rows = []
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append({
                    'stage': row['stage'].strip(),
                    'epoch': int(row['epoch']),
                    'step': int(row['step']),
                    'loss': float(row['loss']),
                    'learning_rate': row.get('learning_rate', ''),
                    'timestamp': row['timestamp']
                })
            except Exception as exc:
                raise ValueError(f'第 {line_no} 行格式错误：{exc}') from exc

    if not rows:
        raise ValueError(f'loss 日志为空：{input_path}')
    return rows


def group_by_stage(rows):
    grouped = defaultdict(list)
    for row in rows:
        stage = row['stage'] or 'unknown'
        grouped[stage].append(row)
    return dict(grouped)


def plot_grouped_loss(grouped, output_path, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError('无法导入 matplotlib，请先安装 matplotlib 后再绘制 loss 曲线。') from exc

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.figure(figsize=(10, 6))
    for stage, rows in grouped.items():
        xs = list(range(1, len(rows) + 1))
        ys = [row['loss'] for row in rows]
        plt.plot(xs, ys, marker='o', linewidth=1.8, markersize=3, label=stage)

    plt.title(title)
    plt.xlabel('Logged step index')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot MiniMind training loss curves.')
    parser.add_argument('--input', default='logs/train_loss.csv', help='loss CSV路径，默认 logs/train_loss.csv')
    parser.add_argument('--output', default='outputs/loss_curve.png', help='总 loss 曲线输出路径')
    parser.add_argument('--output-dir', default='outputs', help='分阶段曲线输出目录')
    parser.add_argument('--stage-prefix', default='', help='分阶段曲线文件名前缀，例如 formal_')
    args = parser.parse_args()

    try:
        rows = load_loss_rows(args.input)
        grouped = group_by_stage(rows)
        plot_grouped_loss(grouped, args.output, 'MiniMind Training Loss')

        for stage, stage_rows in grouped.items():
            safe_stage = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in stage)
            stage_output = os.path.join(args.output_dir, f'{args.stage_prefix}{safe_stage}_loss_curve.png')
            plot_grouped_loss({stage: stage_rows}, stage_output, f'MiniMind {stage} Loss')

        print(f'已保存总 loss 曲线：{args.output}')
        print(f'已保存分阶段 loss 曲线目录：{args.output_dir}')
    except Exception as exc:
        print(f'[plot_loss] {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
