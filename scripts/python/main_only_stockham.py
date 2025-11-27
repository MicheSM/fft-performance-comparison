import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_algorithms(base_dir, algorithm_names):
    data = {}
    for name in algorithm_names:

        # Determine whether it's cooley or stockham
        if "cooley" in name:
            folder = "cooley"
        else:
            folder = "stockham"

        # New file path
        file_path = f"{base_dir}/{folder}/{name}.txt"

        if not os.path.exists(file_path):
            print(f"[WARN] Missing file: {file_path}")
            continue

        data[name] = pd.read_csv(
            file_path,
            sep=' ',
            header=None,
            names=["N", "time1", "time2", "time3", "time4", "time5", "time6", "time7", "time8", "time9"]
        )
    return data


def create_plots(algorithms_data, colors, fmts, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    baseline_alg = 'output_fft_stockham_ci_novector'
    baseline_data = algorithms_data[baseline_alg].groupby('N', as_index=False).mean()

    descriptions = [
        "N",
        "generated input (since last split)",
        "generate input (since start)",
        "initialized vector of roots (since last split)",
        "initialized vector of roots (since start)",
        "bit reversal done (since last split)",
        "bit reversal done (since start)",
        "fft done (since last split)",
        "fft done (since start)",
        "finished (since start)"
    ]

    for checkpoint in range(1, 10):
        plt.figure(figsize=(15, 10))
        baseline_col = baseline_data[f'time{checkpoint}']

        for name, df in algorithms_data.items():
            if name == baseline_alg:
                continue

            alg_avg = df.groupby('N', as_index=False).mean()
            ratio = alg_avg[f'time{checkpoint}'] / baseline_col

            plt.plot(
                alg_avg['N'],
                ratio,
                fmts[name],
                color=colors[name],
                label=name
            )

        plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)

        plt.xlabel('Array size')
        plt.ylabel('Speedup Ratio')
        plt.title(
            f'Performance Relative to {baseline_alg} — {descriptions[checkpoint]}',
            fontsize=14, fontweight='bold'
        )
        plt.xscale('log', base=2)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plot{checkpoint}.png")
        plt.close()


def main():

    algorithm_names = [
        #'output_fft_cooley_bi_novector',
        #'output_fft_cooley_bi_auto',
        #'output_fft_cooley_bi_sve',
        #'output_fft_cooley_bi_sve_3loop',
        #'output_fft_cooley_ci_novector',
        #'output_fft_cooley_ci_auto',
        #'output_fft_cooley_ci_sve',
        #'output_fft_cooley_ci_sve_3loop',
        'output_fft_stockham_bi_novector',
        'output_fft_stockham_bi_auto',
        'output_fft_stockham_bi_sve',
        'output_fft_stockham_bi_sve_2loop',
        'output_fft_stockham_ci_novector',
        'output_fft_stockham_ci_auto',
        'output_fft_stockham_ci_sve'
    ]

    colors = [
        #'skyblue', 'dodgerblue', 'blue', 'darkblue',
        #'lightgray', 'darkgray', 'gray', 'black',
        'lightsalmon', 'red', 'darkred', 'green',
        'yellow', 'gold', 'darkgoldenrod'
    ]

    fmts = [
        #'o--', 'o--', 'o--', 'o--',
        #'o-',  'o-',  'o-',  'o-',
        'o:',  'o:',  'o:', 'o:',
        'o-.', 'o-.', 'o-.'
    ]

    algorithm_colors = {name: c for name, c in zip(algorithm_names, colors)}
    algorithm_fmts   = {name: f for name, f in zip(algorithm_names, fmts)}

    raw_outputs_dir = "data/results/raw-outputs"
    out_plots_dir = "data/results/plots"

    algorithms = load_algorithms(raw_outputs_dir, algorithm_names)

    create_plots(algorithms, algorithm_colors, algorithm_fmts, out_plots_dir)


if __name__ == "__main__":
    main()
