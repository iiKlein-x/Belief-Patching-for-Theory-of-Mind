"""
analyze_belief_localization.py

Answers the research question:
    "Where (layers/tokens) are belief features localized,
     and do they differ between first- and second-order beliefs?"

Inputs:
    --first_order_pkl   : path to causal_STR.pkl for first-order belief dataset
    --second_order_pkl  : path to causal_STR.pkl for second-order belief dataset
    --first_order_jsonl : path to .jsonl predictions file for first-order dataset
    --second_order_jsonl: path to .jsonl predictions file for second-order dataset
    --model_name        : model name string (e.g. gemma2-2B-chat)
    --layer_start       : first layer index that was patched (default 0)
    --output_dir        : where to save plots and results (default: analysis_results/)

Outputs:
    1. layer_localization.png     - mean IE per layer for first vs second order
    2. token_localization.png     - mean IE per token bucket for first vs second order
    3. caf_distribution.png       - per-sample CaF score distributions compared
    4. subject_importance.png     - mean IE on subject tokens vs rest
    5. analysis_results.txt       - all numerical results
"""

import pickle
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, stats
from collections import defaultdict


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(l) for l in f]
    return {d['sample_id']: d for d in data}

def cosine_sim(a, b):
    a, b = np.array(a).flatten(), np.array(b).flatten()
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return None
    return 1. - spatial.distance.cosine(a, b)

def is_valid(ans_score, expl_score, key='diff_prob'):
    """Check that neither matrix is a zero vector and gap is positive."""
    if np.linalg.norm(ans_score[key]) < 1e-8:
        return False
    if np.linalg.norm(expl_score[key]) < 1e-8:
        return False
    gap = ans_score['high_prob'] - ans_score['low_prob']
    if isinstance(gap, (list, np.ndarray)):
        gap = np.mean(gap)
    return gap > 0  # only keep samples where corruption degraded confidence


# ─────────────────────────────────────────────
# Per-sample extraction functions
# ─────────────────────────────────────────────

def get_caf_scores(ans_score, expl_score, key='diff_prob'):
    """Return single, token-wise, and layer-wise CaF for one sample."""
    o = np.maximum(ans_score[key],  0)
    e = np.maximum(expl_score[key], 0)

    # crop to same number of tokens if different (edge case)
    min_t = min(o.shape[0], e.shape[0])
    o, e = o[:min_t], e[:min_t]

    single = cosine_sim(o.flatten(), e.flatten())
    token  = cosine_sim(o.sum(axis=1), e.sum(axis=1))
    layer  = cosine_sim(o.sum(axis=0), e.sum(axis=0))
    return single, token, layer


def get_layer_ie(ans_score, key='diff_prob'):
    """Return per-layer mean IE (summed over token dim, then normalised)."""
    matrix = np.maximum(ans_score[key], 0)   # shape (T, L)
    return matrix.sum(axis=0)                 # shape (L,)


def get_subject_ie(ans_score, key='diff_prob'):
    """
    Return (mean IE on subject tokens, mean IE on non-subject tokens).
    subject_range is stored as [[start, end], [start, end]] for original/cf.
    """
    matrix = np.maximum(ans_score[key], 0)    # (T, L)
    token_ie = matrix.sum(axis=1)             # (T,)

    subj_range = ans_score['subject_range']
    # subject_range[0] = [start, end] for original
    subj_start = 0
    subj_end   = subj_range[0][1] - subj_range[0][0]
    subj_end   = min(subj_end, len(token_ie))

    subj_ie    = token_ie[:subj_end].mean() if subj_end > 0 else 0.
    nonsubj_ie = token_ie[subj_end:].mean()  if subj_end < len(token_ie) else 0.
    return subj_ie, nonsubj_ie


def get_token_buckets(ans_score, pred_dict, key='diff_prob'):
    """
    Divide tokens into 3 buckets:
        subject     - the corrupted subject tokens
        post-subj   - tokens after subject up to midpoint of sequence
        late        - tokens from midpoint onward (question tokens)
    Returns mean IE per bucket.
    """
    matrix   = np.maximum(ans_score[key], 0)
    token_ie = matrix.sum(axis=1)
    T        = len(token_ie)

    subj_range = ans_score['subject_range']
    subj_end   = subj_range[0][1] - subj_range[0][0]
    subj_end   = min(subj_end, T)
    mid        = (subj_end + T) // 2

    subj_bucket      = token_ie[:subj_end].mean()         if subj_end > 0     else 0.
    post_subj_bucket = token_ie[subj_end:mid].mean()      if mid > subj_end   else 0.
    late_bucket      = token_ie[mid:].mean()               if T > mid          else 0.

    return subj_bucket, post_subj_bucket, late_bucket


# ─────────────────────────────────────────────
# Dataset-level aggregation
# ─────────────────────────────────────────────

def aggregate_dataset(pkl_data, jsonl_data, label, layer_start, key='diff_prob'):
    """
    Process all samples in a pkl file and return aggregated results.
    Returns a dict with all per-sample lists ready for analysis.
    """
    results = defaultdict(list)
    skipped = 0

    for sample_id, (ans_score, expl_score) in pkl_data.items():

        if not is_valid(ans_score, expl_score, key):
            skipped += 1
            continue

        # CaF scores
        single, token, layer = get_caf_scores(ans_score, expl_score, key)
        if single is None:
            skipped += 1
            continue
        results['caf_single'].append(single)
        results['caf_token'].append(token)
        results['caf_layer'].append(layer)

        # Layer IE vector
        layer_ie = get_layer_ie(ans_score, key)          # (L,)
        results['layer_ie_ans'].append(layer_ie)

        layer_ie_expl = get_layer_ie(expl_score, key)
        results['layer_ie_expl'].append(layer_ie_expl)

        # Subject vs non-subject IE
        subj_ie, nonsubj_ie = get_subject_ie(ans_score, key)
        results['subj_ie_ans'].append(subj_ie)
        results['nonsubj_ie_ans'].append(nonsubj_ie)

        subj_ie_e, nonsubj_ie_e = get_subject_ie(expl_score, key)
        results['subj_ie_expl'].append(subj_ie_e)
        results['nonsubj_ie_expl'].append(nonsubj_ie_e)

        # Token buckets
        pred_dict = jsonl_data.get(sample_id, {})
        b1, b2, b3 = get_token_buckets(ans_score, pred_dict, key)
        results['bucket_subj'].append(b1)
        results['bucket_post'].append(b2)
        results['bucket_late'].append(b3)

    n_valid = len(results['caf_single'])
    print(f"  [{label}] valid samples: {n_valid}, skipped: {skipped}")
    return results


# ─────────────────────────────────────────────
# Statistical comparison
# ─────────────────────────────────────────────

def compare_distributions(first, second, name, f):
    """Run Mann-Whitney U test and print/write results."""
    if len(first) < 2 or len(second) < 2:
        line = f"{name}: not enough samples for significance test\n"
        print(line, end='')
        f.write(line)
        return

    stat, p = stats.mannwhitneyu(first, second, alternative='two-sided')
    mean_f = np.mean(first)
    mean_s = np.mean(second)
    std_f  = np.std(first)
    std_s  = np.std(second)

    line = (f"{name}:\n"
            f"  First-order:  mean={mean_f:.4f}  std={std_f:.4f}\n"
            f"  Second-order: mean={mean_s:.4f}  std={std_s:.4f}\n"
            f"  Mann-Whitney U={stat:.1f}, p={p:.4f} "
            f"{'*** significant' if p < 0.05 else '(not significant)'}\n\n")
    print(line, end='')
    f.write(line)


# ─────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────

def plot_layer_localization(first_results, second_results, layer_start, output_dir):
    """Plot mean IE per layer for answer and explanation runs, both orders."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for ax, run_key, run_label in zip(
        axes,
        ['layer_ie_ans', 'layer_ie_expl'],
        ['Answer Run', 'Explanation Run']
    ):
        f_layers = np.array(first_results[run_key])   # (N, L)
        s_layers = np.array(second_results[run_key])  # (M, L)

        if len(f_layers) == 0 or len(s_layers) == 0:
            ax.set_title(f'{run_label} — no data')
            continue

        L = min(f_layers.shape[1], s_layers.shape[1])
        f_mean = f_layers[:, :L].mean(axis=0)
        s_mean = s_layers[:, :L].mean(axis=0)
        f_std  = f_layers[:, :L].std(axis=0)
        s_std  = s_layers[:, :L].std(axis=0)
        actual_layers = list(range(layer_start, layer_start + L))

        ax.plot(actual_layers, f_mean, marker='o', label='First-order',
                color='steelblue', linewidth=2)
        ax.fill_between(actual_layers,
                         f_mean - f_std, f_mean + f_std,
                         alpha=0.2, color='steelblue')

        ax.plot(actual_layers, s_mean, marker='s', label='Second-order',
                color='darkorange', linewidth=2)
        ax.fill_between(actual_layers,
                         s_mean - s_std, s_mean + s_std,
                         alpha=0.2, color='darkorange')

        ax.set_xlabel('Layer', fontsize=13)
        ax.set_ylabel('Mean IE score', fontsize=13)
        ax.set_title(f'Layer-wise Belief Localization — {run_label}', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'layer_localization.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_token_buckets(first_results, second_results, output_dir):
    """Bar plot of IE across 3 token buckets for first vs second order."""
    buckets = ['Subject tokens', 'Post-subject tokens', 'Late/question tokens']
    f_means = [np.mean(first_results['bucket_subj']),
               np.mean(first_results['bucket_post']),
               np.mean(first_results['bucket_late'])]
    s_means = [np.mean(second_results['bucket_subj']),
               np.mean(second_results['bucket_post']),
               np.mean(second_results['bucket_late'])]
    f_stds  = [np.std(first_results['bucket_subj']),
               np.std(first_results['bucket_post']),
               np.std(first_results['bucket_late'])]
    s_stds  = [np.std(second_results['bucket_subj']),
               np.std(second_results['bucket_post']),
               np.std(second_results['bucket_late'])]

    x = np.arange(len(buckets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, f_means, w, yerr=f_stds, label='First-order',
           color='steelblue', alpha=0.85, capsize=5)
    ax.bar(x + w/2, s_means, w, yerr=s_stds, label='Second-order',
           color='darkorange', alpha=0.85, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=12)
    ax.set_ylabel('Mean IE score', fontsize=12)
    ax.set_title('Token-wise Belief Localization: First vs Second Order', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'token_localization.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_caf_distributions(first_results, second_results, output_dir):
    """Box plot + strip plot of per-sample CaF scores for both orders."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ['CaF (full)', 'CaF (token-wise)', 'CaF (layer-wise)']
    keys   = ['caf_single', 'caf_token', 'caf_layer']

    for ax, key, label in zip(axes, keys, labels):
        f_vals = first_results[key]
        s_vals = second_results[key]
        data   = [f_vals, s_vals]
        bp = ax.boxplot(data, labels=['First-order', 'Second-order'],
                        patch_artist=True, widths=0.4)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('darkorange')
        bp['boxes'][1].set_alpha(0.7)

        # overlay individual points
        for i, vals in enumerate([f_vals, s_vals], start=1):
            jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       alpha=0.5, s=20, zorder=3,
                       color='steelblue' if i == 1 else 'darkorange')

        ax.set_ylabel('Cosine similarity', fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-1.05, 1.05)

    plt.suptitle('Causal Faithfulness Score Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'caf_distribution.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_subject_importance(first_results, second_results, output_dir):
    """Bar plot: IE on subject tokens vs rest for ans and expl runs."""
    categories = ['Answer\nSubject', 'Answer\nNon-subject',
                  'Expl\nSubject', 'Expl\nNon-subject']
    f_means = [np.mean(first_results['subj_ie_ans']),
               np.mean(first_results['nonsubj_ie_ans']),
               np.mean(first_results['subj_ie_expl']),
               np.mean(first_results['nonsubj_ie_expl'])]
    s_means = [np.mean(second_results['subj_ie_ans']),
               np.mean(second_results['nonsubj_ie_ans']),
               np.mean(second_results['subj_ie_expl']),
               np.mean(second_results['nonsubj_ie_expl'])]

    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, f_means, w, label='First-order',  color='steelblue',  alpha=0.85)
    ax.bar(x + w/2, s_means, w, label='Second-order', color='darkorange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Mean IE score', fontsize=12)
    ax.set_title('Subject Token Importance vs Rest: First vs Second Order', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'subject_importance.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Belief localization analysis: first-order vs second-order ToM"
    )
    parser.add_argument('--first_order_pkl',    required=True,
                        help='Path to causal_STR.pkl for first-order dataset')
    parser.add_argument('--second_order_pkl',   required=True,
                        help='Path to causal_STR.pkl for second-order dataset')
    parser.add_argument('--first_order_jsonl',  required=True,
                        help='Path to .jsonl predictions for first-order dataset')
    parser.add_argument('--second_order_jsonl', required=True,
                        help='Path to .jsonl predictions for second-order dataset')
    parser.add_argument('--model_name',  default='gemma2-2B-chat')
    parser.add_argument('--layer_start', default=0,   type=int,
                        help='First layer index that was patched')
    parser.add_argument('--output_dir',  default='analysis_results',
                        help='Directory to save plots and results')
    parser.add_argument('--key', default='diff_prob',
                        choices=['diff_prob', 'diff_logit',
                                 'normalized_diff_prob', 'normalized_diff_logit'],
                        help='Which IE matrix to use for analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    result_txt = os.path.join(args.output_dir, 'analysis_results.txt')

    print(f"\nLoading first-order data...")
    first_pkl   = load_pkl(args.first_order_pkl)
    first_jsonl = load_jsonl(args.first_order_jsonl)

    print(f"Loading second-order data...")
    second_pkl   = load_pkl(args.second_order_pkl)
    second_jsonl = load_jsonl(args.second_order_jsonl)

    print(f"\nAggregating first-order samples...")
    first_results  = aggregate_dataset(first_pkl,  first_jsonl,
                                        'first-order',  args.layer_start, args.key)

    print(f"Aggregating second-order samples...")
    second_results = aggregate_dataset(second_pkl, second_jsonl,
                                        'second-order', args.layer_start, args.key)

    # ── Plots ──────────────────────────────────
    print(f"\nGenerating plots...")
    plot_layer_localization(first_results, second_results, args.layer_start, args.output_dir)
    plot_token_buckets(first_results, second_results, args.output_dir)
    plot_caf_distributions(first_results, second_results, args.output_dir)
    plot_subject_importance(first_results, second_results, args.output_dir)

    # ── Numerical results ───────────────────────
    print(f"\nWriting numerical results to {result_txt}...")
    with open(result_txt, 'w') as f:
        f.write(f"Belief Localization Analysis\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"IE key: {args.key}\n")
        f.write(f"First-order samples:  {len(first_results['caf_single'])}\n")
        f.write(f"Second-order samples: {len(second_results['caf_single'])}\n\n")
        f.write("=" * 60 + "\n\n")

        # CaF score comparisons
        f.write("── CaF Score Comparisons ──\n\n")
        compare_distributions(first_results['caf_single'],
                               second_results['caf_single'],
                               'CaF (full T×L)', f)
        compare_distributions(first_results['caf_token'],
                               second_results['caf_token'],
                               'CaF (token-wise)', f)
        compare_distributions(first_results['caf_layer'],
                               second_results['caf_layer'],
                               'CaF (layer-wise)', f)

        # Layer localization
        f.write("── Layer Localization ──\n\n")
        if first_results['layer_ie_ans'] and second_results['layer_ie_ans']:
            f_layer = np.array(first_results['layer_ie_ans']).mean(axis=0)
            s_layer = np.array(second_results['layer_ie_ans']).mean(axis=0)
            f_peak  = args.layer_start + int(np.argmax(f_layer))
            s_peak  = args.layer_start + int(np.argmax(s_layer))
            f.write(f"Answer run — First-order  peak layer: {f_peak}\n")
            f.write(f"Answer run — Second-order peak layer: {s_peak}\n\n")

        if first_results['layer_ie_expl'] and second_results['layer_ie_expl']:
            f_layer_e = np.array(first_results['layer_ie_expl']).mean(axis=0)
            s_layer_e = np.array(second_results['layer_ie_expl']).mean(axis=0)
            f_peak_e  = args.layer_start + int(np.argmax(f_layer_e))
            s_peak_e  = args.layer_start + int(np.argmax(s_layer_e))
            f.write(f"Expl run  — First-order  peak layer: {f_peak_e}\n")
            f.write(f"Expl run  — Second-order peak layer: {s_peak_e}\n\n")

        # Token bucket comparisons
        f.write("── Token Bucket Comparisons (Answer Run) ──\n\n")
        compare_distributions(first_results['bucket_subj'],
                               second_results['bucket_subj'],
                               'Subject token IE', f)
        compare_distributions(first_results['bucket_post'],
                               second_results['bucket_post'],
                               'Post-subject token IE', f)
        compare_distributions(first_results['bucket_late'],
                               second_results['bucket_late'],
                               'Late/question token IE', f)

        # Subject importance
        f.write("── Subject Token Importance ──\n\n")
        compare_distributions(first_results['subj_ie_ans'],
                               second_results['subj_ie_ans'],
                               'Subject IE — Answer run', f)
        compare_distributions(first_results['subj_ie_expl'],
                               second_results['subj_ie_expl'],
                               'Subject IE — Explanation run', f)

    print(f"\nDone. All outputs saved to: {args.output_dir}/")
    print(f"  layer_localization.png")
    print(f"  token_localization.png")
    print(f"  caf_distribution.png")
    print(f"  subject_importance.png")
    print(f"  analysis_results.txt")


if __name__ == '__main__':
    main()
