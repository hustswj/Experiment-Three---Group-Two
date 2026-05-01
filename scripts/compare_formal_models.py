#!/usr/bin/env python3
import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)

import compare_models


FORMAL_DEFAULTS = [
    '--official-transformers-path', 'minimind-3',
    '--official-weight', 'full_sft',
    '--official-hidden-size', '768',
    '--official-num-hidden-layers', '8',
    '--self-weight', 'formal_full_sft',
    '--self-hidden-size', '768',
    '--self-num-hidden-layers', '8',
    '--finetune-base-weight', 'formal_full_sft',
    '--finetune-lora-weight', 'formal_lora_hust',
    '--finetune-hidden-size', '768',
    '--finetune-num-hidden-layers', '8',
    '--raw-output', os.path.join(ROOT_DIR, 'outputs/formal_model_compare_raw.jsonl'),
    '--markdown-output', os.path.join(ROOT_DIR, 'outputs/formal_model_compare.md'),
]


def main():
    sys.argv = [sys.argv[0]] + FORMAL_DEFAULTS + sys.argv[1:]
    compare_models.main()


if __name__ == '__main__':
    main()
