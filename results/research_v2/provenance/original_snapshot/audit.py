"""Сверить итоговые результаты с весами, данными и сохранёнными предсказаниями."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment import OUT, ROOT, load_data, metadata, read_checkpoint, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = json.loads((OUT/'frozen_selection.json').read_text())
    evaluation = json.loads((OUT/'test_evaluation.json').read_text())
    assert evaluation['completed']
    meta, cat = metadata()
    data = load_data('test')
    checked = []
    for filename, expected in meta['prepared_sha256'].items():
        assert sha(OUT/filename) == expected
    for filename, expected in frozen['source_sha256'].items():
        assert sha(ROOT/filename) == expected
    for name, expected in frozen['checkpoint_sha256'].items():
        assert sha(OUT/name/'best.pt') == expected
        state = read_checkpoint(OUT/name/'best.pt')
        expected_steps = 1000 if name.startswith('rl_') else (frozen['extra_sft_steps'] if name.startswith('sft_') else 6000)
        assert state['step'] == expected_steps
    assert sha(OUT/'reward_s11'/'best.pt') == frozen['reward_checkpoint_sha256']
    assert sha(OUT/'reward_s22'/'best.pt') == frozen['auditor_checkpoint_sha256']
    for name, metrics in evaluation['models'].items():
        with np.load(OUT/name/'test_predictions.npz') as p:
            top = p['top']
            assert np.array_equal(p['uid'],data['uid'])
            assert top.shape == (len(data['target']),20)
            assert np.all(top >= 2)
            assert np.all(cat['first_seen'][top] < data['timestamp'][:,None])
            assert all(len(set(row)) == 20 for row in top)
            for k in [10,20]:
                recalls, ndcgs = [], []
                for row, target in zip(top,data['target']):
                    places = np.flatnonzero(row[:k] == target)
                    recalls.append(float(bool(len(places))))
                    ndcgs.append(1/np.log2(int(places[0])+2) if len(places) else 0.)
                # Независимое словарное агрегирование вместо bincount.
                for metric, values in [(f'recall@{k}',recalls),(f'ndcg@{k}',ndcgs)]:
                    by_user = {}
                    for uid,value in zip(data['uid'],values):
                        by_user.setdefault(int(uid),[]).append(value)
                    actual = float(np.mean([np.mean(v) for v in by_user.values()]))
                    assert abs(actual-metrics[metric]) < 1e-10
                    assert np.allclose(values,p[metric],atol=1e-12)
        checked.append(name)
    # RL checkpoints действительно отличаются от соответствующего NIP.
    parameter_changes = {}
    for name in frozen['models']:
        if name.startswith('rl_'):
            seed = name.rsplit('_s',1)[1]
            actor = read_checkpoint(OUT/name/'best.pt')['model']
            reference = read_checkpoint(OUT/f'baseline_s{seed}'/'best.pt')['model']
            delta = max(float((actor[k]-reference[k]).abs().max()) for k in actor)
            assert delta > 0
            parameter_changes[name] = delta
    for rm in ['reward_s11','reward_s22']:
        quality = json.loads((OUT/rm/'test.json').read_text())
        assert quality['reward_model']['n'] > 0
        assert quality['calibration']
    for name in frozen['models']:
        assert len(evaluation['models'][name]['rollout']) == 5
    result = {'status':'passed', 'test_prediction_models_checked':checked,
              'metrics_independently_recomputed':['Recall@10','Recall@20','NDCG@10','NDCG@20'],
              'data_hashes':'passed','source_hashes':'passed','checkpoint_hashes':'passed',
              'future_candidates':'passed','rl_max_parameter_changes':parameter_changes,
              'reward_test_quality':'present','five_step_rollouts':'present'}
    write_json(OUT/'completion_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
