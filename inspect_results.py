import pickle
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--dataset_name', type=str, default='mydataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='STR')
    args = parser.parse_args()

    pkl_path = f'prediction/{args.model_name}/{args.dataset_name}/{args.seed}/causal_{args.mode}.pkl'
    print(f'Loading from: {pkl_path}\n')

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    for sample_id, (ans_result, expl_result) in results.items():
        print(f"Sample {sample_id}")
        print(f"Answer IE matrix shape:      {ans_result['diff_prob'].shape}")
        print(f"Explanation IE matrix shape: {expl_result['diff_prob'].shape}")
        print(f"Answer tokens:               {ans_result['input_tokens'][0]}")
        print(f"Subject range:               {ans_result['subject_range']}")
        print(f"\n--- ANSWER RUN ---")
        print(f"diff_prob shape: {ans_result['diff_prob'].shape}")
        print(f"diff_prob (patched - corrupted):\n{ans_result['diff_prob']}")
        print(f"diff_logit shape: {ans_result['diff_logit'].shape}")
        print(f"diff_logit (patched - corrupted):\n{ans_result['diff_logit']}")
        print(f"normalized_diff_prob:\n{ans_result['normalized_diff_prob']}")
        
        
        print(f"\n--- EXPLANATION RUN ---")
        print(f"diff_prob shape: {expl_result['diff_prob'].shape}")
        print(f"diff_prob (patched - corrupted):\n{expl_result['diff_prob']}")
        print(f"diff_logit shape: {expl_result['diff_logit'].shape}")
        print(f"diff_logit (patched - corrupted):\n{expl_result['diff_logit']}")
        print(f"normalized_diff_prob:\n{expl_result['normalized_diff_prob']}")
        
        

        
        print(f"\n--- REFERENCE SCORES ---")
        print(f"high_prob: {ans_result['high_prob']}")
        print(f"low_prob:  {ans_result['low_prob']}")
        print(f"high_logit: {ans_result['high_logit']}")
        print(f"low_logit:  {ans_result['low_logit']}")
        print()

if __name__ == '__main__':
    main()