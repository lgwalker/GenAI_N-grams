import json
import pickle
import math
import argparse
import sys

'''Test script that takes argument of any test file to be used on the trained model
that was found in the Training.ipynb'''

def load_methods(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]

def generate_json_results(input_file, output_file, model, vocab):
    methods = load_methods(input_file)

    n = model.order
    json_data = []
    total_log_prob = 0
    total_token_count = 0

    for i, tokens in enumerate(methods):
        # Map to vocab
        processed_tokens = [t if t in vocab else "<UNK>" for t in tokens]
        
        # Padding (n-1) start symbols to predict the first token
        padded = ["<s>"] * (n - 1) + processed_tokens + ["</s>"]
        
        method_predictions = []
        
        # Sliding window for predicitons
        for j in range(n - 1, len(padded)):
            context = tuple(padded[j-(n-1):j])
            ground_truth = padded[j]
            
            # Probability of the actual token and prediction
            # logscore log2(P) for perplexity
            log_prob = model.logscore(ground_truth, context)
            total_log_prob += log_prob
            total_token_count += 1
            
            # Find the predicted token
            pred_token = model.generate(1, text_seed=context)
            pred_prob = model.score(pred_token, context)
            
            method_predictions.append({
                "context": list(context),
                "predToken": pred_token,
                "predProbability": round(pred_prob, 4),
                "groundTruth": ground_truth
            })

        json_data.append({
            "index": f"ID{i+1}",
            "tokenizedCode": " ".join(tokens),
            "contextWindow": n,
            "predictions": method_predictions
        })

    # Final perplexity calculation
    avg_log_prob = total_log_prob / total_token_count if total_token_count > 0 else 0
    final_perplexity = 2 ** (-avg_log_prob)

    output_json = {
        "testSet": input_file,
        "perplexity": round(final_perplexity, 2),
        "data": json_data
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4)
    
    print(f"Generated {output_file} | Perplexity: {final_perplexity:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Generate N-gram model results for Java methods.")
    
    # Define the arguments (optional if testing with other files from command line)
    parser.add_argument("--test_file", type=str, default="./ngram_dataset/test.txt", help="Path to the provided test set.")
    parser.add_argument("--test_output", type=str, default="results-xxxxxx.json", help="Output JSON for provided test set.")
    parser.add_argument("--mined_file", type=str, default="./ngram_dataset/test_self_mined.txt", help="Path to self-mined test set.")
    parser.add_argument("--mined_output", type=str, default="results-yyyyyy.json", help="Output JSON for self-mined test set.")

    args = parser.parse_args()

    # Load model and vocab
    try:
        with open("./best_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open("./best_vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find model or vocab files. {e}")
        sys.exit(1)

    print(f"Testing the model with the {args.test_file}")
    generate_json_results(args.test_file, args.test_output, model, vocab)
    
    print(f"Testing the model with the {args.mined_output}")
    generate_json_results(args.mined_file, args.mined_output, model, vocab)

if __name__ == "__main__":
    main()