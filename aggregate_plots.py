import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

from pathlib import Path



name_dict = {
    'reward_bench_gen': 'RewardBench-Generative',
    'auto_j': 'Auto-J',
    'instrusum': 'InstruSum',
    'hhh': 'HHH',
    'preference_bench': 'PreferenceBench',
    'eval_bias_bench': 'EvalBiasBench',
    'lfqa_eval': 'LFQA'
}



def load_json_files(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data.append(json.load(file))
    return data

def prepare_data(json_data):
    # Assuming each JSON file has the same structure
    benchmarks = list(json_data[0].keys())
    scores = [json_data[k]['overall'] for k in benchmarks]
    benchmarks = [name_dict[k] for k in benchmarks]
    # scores = [list(d.values()) for d in json_data]
    return benchmarks, scores

def radar_plot(benchmarks, scores, trial_labels):
    num_vars = len(benchmarks)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for trial_scores, label in zip(scores, trial_labels):
        trial_scores += trial_scores[:1]  # complete the loop
        ax.plot(angles, trial_scores, linewidth=1, linestyle='solid', label=label)
        ax.fill(angles, trial_scores, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(benchmarks)
    ax.set_title("Radar Plot of Trials")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.show()

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate a radar plot from multiple JSON files.")
    parser.add_argument('result_paths', nargs='+', help='Paths to JSON files containing aggregated eval results')

    args = parser.parse_args()

    # Load and process JSON files
    json_data = load_json_files(args.json_files)
    benchmarks, scores = prepare_data(json_data)
    trial_labels = [Path(file).stem for file in args.json_files]

    # Plot radar chart
    radar_plot(benchmarks, scores, trial_labels)

if __name__ == "__main__":
    main()
