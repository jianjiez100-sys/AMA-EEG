"""Compare retained source computations against the original experiments.

SEED dynamic validation deliberately uses public-release CE alpha instead of
source entropy, so its validation policy is checked separately in protocol tests.
"""
import argparse
import ast
import contextlib
import copy
import gc
import io
import logging
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from torch import nn
from torch.nn import functional as F
from transformers.models.bert import BertConfig, BertModel

from data.dataset import PretrainSampler
from data.feature_processing import normalize_extraction_input, smooth_trial_features
from model import get_extractor_class
from model.models import simpleNN3


def definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f'Missing definitions in {path}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def compare(source_dir, dataset):
    source = Path(source_dir)
    namespace = dict(torch=torch, nn=nn, F=F, np=np, pl=pl,
                     BertConfig=BertConfig, BertModel=BertModel,
                     random=random, logging=logging, gc=gc,
                     log=logging.getLogger(__name__))
    definitions(source / 'model/models.py',
                ['ResidualAdd', 'stratified_layerNorm', 'subject_aware_norm',
                 'Conv_att_simple_new', 'simpleNN3'], namespace)
    definitions(source / 'model/loss/con_loss.py', ['MultiModalInfoNCELoss'], namespace)
    definitions(source / 'model/pl_models.py', ['compute_entropy', 'ExtractorModel'], namespace)
    definitions(source / 'data/dataset.py', ['PretrainSampler'], namespace)
    definitions(source / 'data/data_process.py', ['LDS'], namespace)
    with initialize_config_dir(config_dir=str(ROOT / 'cfgs'), version_base='1.3'):
        cfg = compose(config_name='config', overrides=[f'data={dataset}'])
    cfg.model.use_ln_backbone = True
    for modality in ('text', 'image'):
        key = f'pretrained_{modality}_proj'
        cfg.train[key] = str(ROOT / cfg.train[key])
    kwargs = dict(cfg.model)
    kwargs.pop('_target_')
    if dataset == 'SEED':
        kwargs.pop('proj_type')
    torch.manual_seed(7)
    original = namespace['ExtractorModel'](namespace['Conv_att_simple_new'](**kwargs), cfg.train)
    torch.manual_seed(7)
    release = get_extractor_class(dataset)(hydra.utils.instantiate(cfg.model), cfg.train)
    release.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
    generator = torch.Generator().manual_seed(23)
    n_videos = cfg.data.n_vids
    eeg = torch.randn(2 * n_videos, 1, cfg.data.n_channs, 625, generator=generator)
    text = torch.randn(2 * n_videos, 5, 1024, generator=generator)
    image = torch.randn(2 * n_videos, 5, 1024, generator=generator)
    labels = torch.arange(n_videos).remainder(cfg.data.n_class).repeat(2)
    ids = torch.arange(n_videos).repeat(2)
    subjects = torch.arange(2).repeat_interleave(n_videos)
    extras = (subjects * 100000 + ids * 1000, ids * 1000) if dataset == 'SEED' else (ids, subjects)
    batch = (eeg, labels, text, image, *extras)
    logged = []
    losses, gradients = [], []
    for module in (original, release):
        metrics = {}
        module.log = lambda name, value, **kw: metrics.update({name: value.detach().clone() if torch.is_tensor(value) else value})
        module.log_dict = lambda values, **kw: metrics.update({k: v.detach().clone() if torch.is_tensor(v) else v for k, v in values.items()})
        module.optimizers = lambda: type('Optimizer', (), {'param_groups': [{'lr': cfg.train.lr}]})()
        module.train()
        torch.manual_seed(29)
        loss = module.training_step(batch, 0)
        loss.backward()
        losses.append(loss.detach())
        gradients.append({name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()
                          if parameter.grad is not None})
        module.eval()
        with torch.no_grad():
            if dataset == 'FACED':
                module.validation_step(batch, 0)
            module.model.set_stratified([])
            metrics['backbone'] = module.predict_step((eeg[:4], labels[:4]), 0)
        logged.append(metrics)
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    if gradients[0].keys() != gradients[1].keys():
        raise AssertionError('Gradient keys differ')
    max_gradient_error = 0.0
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
        max_gradient_error = max(max_gradient_error, (gradients[0][name] - gradients[1][name]).abs().max().item())
    common_metrics = sorted(logged[0].keys() & logged[1].keys())
    for key in common_metrics:
        if torch.is_tensor(logged[0][key]):
            torch.testing.assert_close(logged[0][key], logged[1][key], rtol=0, atol=0)
    counts = np.full((cfg.data.n_session, n_videos), 4)
    sequences = []
    for sampler_class in (namespace['PretrainSampler'], PretrainSampler):
        random.seed(7)
        np.random.seed(7)
        sequences.append(list(sampler_class(3, n_videos, counts, cfg.train.sampler_times)))
    assert len(sequences[0]) == len(sequences[1])
    for original_batch, release_batch in zip(*sequences):
        torch.testing.assert_close(original_batch, release_batch, rtol=0, atol=0)
    torch.manual_seed(7)
    original_mlp = namespace['simpleNN3'](1024, [512, 128], cfg.data.n_class, 0.2, 'no')
    torch.manual_seed(7)
    release_mlp = simpleNN3(1024, [512, 128], cfg.data.n_class, 0.2, 'no')
    original_mlp.eval()
    release_mlp.eval()
    with torch.no_grad():
        torch.testing.assert_close(original_mlp(text[:4].mean(1)), release_mlp(text[:4].mean(1)), rtol=0, atol=0)
    if dataset == 'FACED':
        definitions(source / 'ext_fea_reorder.py', ['normTrain', 'process_sub_LDS'], namespace)
    else:
        definitions(source / 'extract_fea.py', ['process_sub_LDS'], namespace)
    small = np.random.default_rng(7).normal(size=(3, 6, 4, 9)).astype(np.float32)
    if dataset == 'FACED':
        expected = namespace['normTrain'](small, small[[1, 2]])
    else:
        # Execute the actual source fast-path statements, with fixture globals.
        tree = ast.parse((source / 'extract_fea.py').read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ext_fea')
        precompute = next(node for node in main.body if isinstance(node, ast.If)
                          and ast.unparse(node.test) == 'cfg.ext_fea.normTrain')
        loop = next(node for node in main.body if isinstance(node, ast.For) and ast.unparse(node.target) == 'fold')
        normalization = next(node for node in loop.body if isinstance(node, ast.If)
                             and ast.unparse(node.test) == 'cfg.ext_fea.normTrain and precompute_done')
        namespace.update(cfg=cfg, data2=small, data2_fold=np.empty_like(small),
                         train_subs=[1, 2], time=__import__('time'),
                         tqdm=lambda iterable, **kw: iterable)
        cfg.data.n_subs = 3
        for node in (precompute, normalization):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<source fast path>', 'exec'), namespace)
        expected = namespace['data2_fold']
    np.testing.assert_array_equal(expected, normalize_extraction_input(small, [1, 2], dataset))
    features = np.random.default_rng(8).normal(size=(3, 6, 8)).astype(np.float32)
    trial_counts = np.array([2, 4])
    boundaries = np.array([0, 2, 6])
    original_smoothed = np.stack([namespace['process_sub_LDS'](i, row, trial_counts, boundaries)[1]
                                  for i, row in enumerate(features)])
    np.testing.assert_array_equal(original_smoothed, smooth_trial_features(features, trial_counts))
    validation = 'source match' if dataset == 'FACED' else 'CE policy checked separately'
    print(f'{dataset} (dynamic; validation: {validation}): exact loss={losses[0].item():.8f}, gradients={len(gradients[0])}, '
          f'max gradient error={max_gradient_error}, common metrics={len(common_metrics)}, '
          f'sampler batches={len(sequences[0])}; backbone, MLP, normalization, LDS match')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--faced-source', type=Path, required=True)
    parser.add_argument('--seed-source', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for source, dataset in ((args.faced_source, 'FACED'), (args.seed_source, 'SEED')):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            compare(source, dataset)
        print(output.getvalue().splitlines()[-1])


if __name__ == '__main__':
    main()
