"""Probe saved Adapter inputs through current generation and the real CLIP teacher.

Generation-only diagnostic, not a replay of missing failed-batch RNG/risk state
and not an accuracy/privacy experiment. Source experiment files are read-only.
"""
import argparse
import io
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import yaml
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from privacy_defenses.risk_synthesis import LocalGeometry, RiskSynthesis
from trainmodel.clip_adapter import build_clip_adapter_text_features


@torch.no_grad()
def probe(run, snapshot, client, sample, risk, seed):
    run = Path(run)
    config = yaml.safe_load((run / 'run_config.yaml').read_text())
    if config['model_type'] != 'clip_adapter':
        raise ValueError('This probe uses the Adapter class-text template.')
    options = dict(config['defense']['synthesis'])
    for key in ('norm_ratio_min', 'norm_ratio_max'):
        options.pop(key, None)
    if (options['views_per_record'] != 2 or options['candidate_selection'] != 'first_semantic'
            or not options['semantic_filter'] or options['risk_history'] != 'none'):
        raise ValueError('Probe requires the current two-view semantic validation protocol.')
    root = run / 'risk_synthesis'
    def read(name):
        return torch.load(root / name, map_location='cpu', weights_only=True, mmap=True)
    state = read(f'client_{client}_distribution.pt')
    geometry = LocalGeometry.__new__(LocalGeometry)
    geometry.codes = read(f'client_{client}_source_codes.pt')
    geometry.labels, geometry.classes = state['labels'], state['classes']
    geometry.global_distribution = read('global_distribution.pt')
    teacher = CLIPModel.from_pretrained(snapshot, local_files_only=True).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    meta = next((ROOT / 'data/CIFAR100/data').rglob('meta'))
    names = pickle.loads(meta.read_bytes(), encoding='latin1')['fine_label_names']
    synth = RiskSynthesis({'synthesis': options}, config['seed'])
    synth.teacher = teacher
    synth.text = build_clip_adapter_text_features(teacher, processor, names, 'cifar100', torch.device('cpu'))
    synth.geometry[client] = geometry
    synth.generators[client] = torch.Generator().manual_seed(seed)
    synth.handle = io.StringIO()
    emb = teacher.vision_model.embeddings
    cls = emb.class_embedding + emb.position_embedding.weight[0]
    original = torch.cat((cls[None], geometry.codes[sample].reshape(49, 768)))[None]
    model = SimpleNamespace(encode_input_tokens=lambda images: original.clone())
    labels = geometry.labels[sample:sample+1]
    views = synth.transform_views(model, SimpleNamespace(id=client), torch.zeros(1), labels,
                                  torch.tensor([sample]), torch.tensor([risk]), 1, 0, 0)
    records = [group[0] for group in synth.pending_views[client]]
    assert all(torch.isfinite(v).all() and not torch.equal(v, original) for v in views)
    assert not torch.equal(views[0], views[1])
    assert not synth.counts  # No optimizer step, so no committed training exposure.
    return dict(scope='generation_only_with_real_frozen_teacher; not failed-RNG replay or training-effect evidence',
                source_run=str(run.resolve()), client=client, sample_id=sample, label=int(labels[0]),
                probe_risk=risk, probe_seed=seed, implementation=synth.summary()['implementation'],
                source_norm=float(geometry.codes[sample].double().norm()),
                global_mean_norm=float(geometry.global_class_center(sample).double().norm()),
                finite_changed_distinct_views=True, original_fallback=False, optimizer_steps=0,
                below_previous_lower_bound=sum(r['norm_ratio'] < .1 for r in records), views=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--client', type=int, default=3)
    parser.add_argument('--sample', type=int, default=440)
    parser.add_argument('--risk', type=float, default=25.5/26)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = probe(args.run, args.snapshot, args.client, args.sample, args.risk, args.seed)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x') as handle:
            handle.write(encoded + '\n')
    print(encoded)
