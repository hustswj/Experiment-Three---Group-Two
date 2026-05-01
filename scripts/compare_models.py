#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_model_params, setup_seed


PROMPTS = [
    "请用通俗语言解释什么是反向传播。",
    "从数据准备到部署，概括大语言模型训练的主要流程。",
    "监督学习和无监督学习有什么区别？请举一个简单例子。",
    "研究生刚入学时应该如何规划科研训练？请给出三点建议。",
    "请将下面这段话总结成不超过60字：深度学习实验通常包括阅读代码、准备数据、运行训练、记录损失、保存模型、设计测试问题和分析模型输出。小规模实验虽然不能代表最终效果，但可以帮助确认训练链路是否正确。",
    "小明有3本书，小红给了他2本，他又送给同学1本。小明现在有几本书？请给出计算过程。",
    "根据 MiniMind 项目的常见流程，pretrain、SFT 和 LoRA 分别起什么作用？",
    "如果要制定华中科技大学研究生阶段的培养计划，通常应关注哪些方面？请只给一般性建议。"
]


@dataclass
class ModelSpec:
    name: str
    kind: str
    load_from: str
    weight: str = ''
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    lora_weight: str = ''
    note: str = ''


def native_weight_path(save_dir, weight, hidden_size, use_moe=False):
    moe_suffix = '_moe' if use_moe else ''
    return os.path.join(ROOT_DIR, save_dir, f'{weight}_{hidden_size}{moe_suffix}.pth')


def build_model_specs(args):
    specs = []
    skipped = []

    official_tf_path = os.path.join(ROOT_DIR, args.official_transformers_path)
    if os.path.isdir(official_tf_path) and os.path.exists(os.path.join(official_tf_path, 'config.json')):
        specs.append(ModelSpec(
            name='official_transformers_minimind_3',
            kind='transformers',
            load_from=official_tf_path,
            note='本地 transformers 格式官方模型目录'
        ))
    else:
        skipped.append(f'未找到官方 transformers 模型目录：{args.official_transformers_path}')

    official_native = native_weight_path(args.save_dir, args.official_weight, args.official_hidden_size)
    if os.path.exists(official_native):
        specs.append(ModelSpec(
            name=f'official_or_existing_{args.official_weight}_{args.official_hidden_size}',
            kind='native',
            load_from=os.path.join(ROOT_DIR, 'model'),
            weight=args.official_weight,
            hidden_size=args.official_hidden_size,
            num_hidden_layers=args.official_num_hidden_layers,
            note='本地 out 目录中已有的 full_sft 权重，可能是官方权重或用户预先放置的权重'
        ))
    else:
        skipped.append(f'未找到官方/现有 PyTorch 权重：{official_native}')

    self_weight = native_weight_path(args.save_dir, args.self_weight, args.self_hidden_size)
    if os.path.exists(self_weight):
        specs.append(ModelSpec(
            name=f'self_trained_{args.self_weight}_{args.self_hidden_size}',
            kind='native',
            load_from=os.path.join(ROOT_DIR, 'model'),
            weight=args.self_weight,
            hidden_size=args.self_hidden_size,
            num_hidden_layers=args.self_num_hidden_layers,
            note='本次课程烟测训练得到的自训练模型'
        ))
    else:
        skipped.append(f'未找到自训练 PyTorch 权重：{self_weight}')

    lora_path = native_weight_path(args.save_dir, args.finetune_lora_weight, args.finetune_hidden_size)
    lora_base_path = native_weight_path(args.save_dir, args.finetune_base_weight, args.finetune_hidden_size)
    if os.path.exists(lora_path) and os.path.exists(lora_base_path):
        specs.append(ModelSpec(
            name=f'finetuned_{args.finetune_lora_weight}_on_{args.finetune_base_weight}_{args.finetune_hidden_size}',
            kind='native',
            load_from=os.path.join(ROOT_DIR, 'model'),
            weight=args.finetune_base_weight,
            hidden_size=args.finetune_hidden_size,
            num_hidden_layers=args.finetune_num_hidden_layers,
            lora_weight=args.finetune_lora_weight,
            note='检测到的 LoRA 微调模型'
        ))
    else:
        skipped.append(f'未找到微调 LoRA 权重或基座权重：{lora_path} / {lora_base_path}')

    return specs, skipped


def load_model(spec, args):
    tokenizer = AutoTokenizer.from_pretrained(spec.load_from)
    device = torch.device(args.device)

    if spec.kind == 'transformers':
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(spec.load_from, torch_dtype=dtype, trust_remote_code=True)
        model = model.eval().to(device)
        get_model_params(model, model.config)
        return model, tokenizer

    lm_config = MiniMindConfig(
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_hidden_layers,
        use_moe=spec.use_moe,
    )
    model = MiniMindForCausalLM(lm_config)
    weight_path = native_weight_path(args.save_dir, spec.weight, spec.hidden_size, spec.use_moe)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    if spec.lora_weight:
        apply_lora(model)
        lora_path = native_weight_path(args.save_dir, spec.lora_weight, spec.hidden_size, spec.use_moe)
        load_lora(model, lora_path)

    model = model.eval().to(device)
    if device.type == 'cuda':
        model = model.half()
    get_model_params(model, model.config)
    return model, tokenizer


def generate_answer(model, tokenizer, prompt, spec, args):
    if 'pretrain' in spec.weight:
        input_text = tokenizer.bos_token + prompt
    else:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=bool(args.open_thinking)
        )

    inputs = tokenizer(input_text, return_tensors='pt', truncation=True).to(args.device)
    setup_seed(args.seed)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=max(args.temperature, 1e-6),
            repetition_penalty=args.repetition_penalty
        )
    new_tokens = generated_ids[0][len(inputs['input_ids'][0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def md_escape(text, limit=None):
    text = text.replace('\r', ' ').replace('\n', '<br>')
    text = text.replace('|', '\\|')
    if limit and len(text) > limit:
        return text[:limit] + '...'
    return text


def write_outputs(results, skipped, args):
    os.makedirs(os.path.dirname(args.raw_output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.markdown_output) or '.', exist_ok=True)

    with open(args.raw_output, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(args.markdown_output, 'w', encoding='utf-8') as f:
        f.write('# MiniMind 模型生成效果对比\n\n')
        f.write(f'生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('## 生成配置\n\n')
        f.write(f'- max_new_tokens：{args.max_new_tokens}\n')
        f.write(f'- temperature：{args.temperature}\n')
        f.write(f'- top_p：{args.top_p}\n')
        f.write(f'- repetition_penalty：{args.repetition_penalty}\n')
        f.write(f'- device：{args.device}\n\n')

        if skipped:
            f.write('## 跳过的模型\n\n')
            for item in skipped:
                f.write(f'- {item}\n')
            f.write('\n')

        if not results:
            f.write('## 对比结果\n\n')
            f.write('未检测到可用于推理的模型权重，因此没有生成回答。\n')
            return

        f.write('## 对比结果\n\n')
        f.write('表格中的回答为真实推理输出；较长内容会截断展示，完整文本见 `outputs/model_compare_raw.jsonl`。\n\n')
        f.write('| 问题 | 模型 | 生成回答 |\n')
        f.write('|---|---|---|\n')
        for item in results:
            f.write(
                f"| {md_escape(item['prompt'])} | {md_escape(item['model_name'])} | "
                f"{md_escape(item['generated_answer'], args.table_answer_chars)} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description='Compare MiniMind model generations on fixed Chinese prompts.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='out')
    parser.add_argument('--official-transformers-path', default='minimind-3')
    parser.add_argument('--official-weight', default='full_sft')
    parser.add_argument('--official-hidden-size', type=int, default=768)
    parser.add_argument('--official-num-hidden-layers', type=int, default=8)
    parser.add_argument('--self-weight', default='course_sft')
    parser.add_argument('--self-hidden-size', type=int, default=64)
    parser.add_argument('--self-num-hidden-layers', type=int, default=2)
    parser.add_argument('--finetune-base-weight', default='course_sft')
    parser.add_argument('--finetune-lora-weight', default='lora_hust')
    parser.add_argument('--finetune-hidden-size', type=int, default=64)
    parser.add_argument('--finetune-num-hidden-layers', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--repetition-penalty', type=float, default=1.05)
    parser.add_argument('--open-thinking', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raw-output', default='outputs/model_compare_raw.jsonl')
    parser.add_argument('--markdown-output', default='outputs/model_compare.md')
    parser.add_argument('--table-answer-chars', type=int, default=300)
    args = parser.parse_args()

    specs, skipped = build_model_specs(args)
    results = []
    generation_config = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'open_thinking': bool(args.open_thinking),
        'seed': args.seed
    }

    for spec in specs:
        print(f'[compare] 加载模型：{spec.name}')
        try:
            model, tokenizer = load_model(spec, args)
        except Exception as exc:
            skipped.append(f'{spec.name} 加载失败：{exc}')
            continue

        for prompt in PROMPTS:
            print(f'[compare] {spec.name} <- {prompt[:24]}...')
            try:
                answer = generate_answer(model, tokenizer, prompt, spec, args)
            except Exception as exc:
                answer = f'[生成失败] {exc}'
            results.append({
                'prompt': prompt,
                'model_name': spec.name,
                'generated_answer': answer,
                'generation_config': generation_config,
                'model_note': spec.note
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(results, skipped, args)
    print(f'[compare] 原始结果：{args.raw_output}')
    print(f'[compare] Markdown 表格：{args.markdown_output}')


if __name__ == '__main__':
    random.seed(42)
    main()
