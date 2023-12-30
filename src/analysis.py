import os
import math
import pandas as pd
from matplotlib.cm import get_cmap
from tabulate import tabulate

from utils.config import *
from utils.Filter import Filter
from utils.Dataset import Dataset, save, load
from utils.Summary import Summary
from utils.Helper import create_directory, print_divider, process_run
from utils.Protocol import Protocol

class Sweep:
    def __init__(self, kwargs, filter_params, sparams, vc_colors, adv_colors):
        self.filter_params = filter_params
        self.kwargs = kwargs
        self.sparams = sparams
        self.vc_colors = vc_colors
        self.adv_colors = adv_colors
        self.table = []
        self.color_index = 0

    def execute(self):
        self.initialize_sweep()
        print("Security parameters:", self.sparams)
        self.execute_sweep()
        self.finalize_sweep()

    def initialize_sweep(self):
        self.filter_update = [x for x in self.filter_params if self.kwargs[x]]
        self.initialize_filter()

    def finalize_sweep(self):
        print(tabulate(self.table, headers='firstrow', tablefmt='fancy_grid'))
        print_divider()
    
    def initialize_filter(self):
        if self.kwargs["filter"] or self.kwargs["initialize"]:
            filter_args = {
                "dataset": load(DATASET_PKL),
                "split": None,
                "iou": True,
                "horizon": True,
                "theta": THETA if not self.kwargs["theta"] else iter_info['theta'][0],
                "visible_horizon": VISIBLE,
                "iou_threshold": IOU_THRESHOLD,
                "claimed_distance_delta": CLAIMED_DISTANCE,
                "x_px_interval": X_PX_INTERVAL
            }
            if self.filter_update:
                # Update filter_args based on kwargs and iter_info
                pass  # Add logic to update filter_args as needed

            self.filter = Filter(**filter_args)
            save(self.filter, FILTERED_DATASET_PKL)
        else:
            self.filter = load(FILTERED_DATASET_PKL)

    def execute_sweep(self):
        index = 0
        sweep_options = ["ab", "scoring", "threshold", "kernel", "rate", "visible", "dist"]
        for key in sweep_options:
            if self.kwargs[key]:
                column_names = list(self.sparams.keys()) + ["Passing Rate", "Adversary Passing Rate", "E(VC)", "E(A)",
                                                            "SD(VC)", "SD(A)"]
                self.table = [column_names]
                self.df = pd.DataFrame(columns=column_names)
                colors = (self.vc_colors[self.color_index], self.adv_colors[self.color_index])
                self.unified_iterate(key)
                if key == "ab":
                    break
                self.color_index += 1

    def unified_iterate(self, key):
        if key == 'ab':
            for a in iter_info['threshold']:
                self.sparams["alpha"] = a
                for i in iter_info['scoring']:
                    self.sparams["beta"] = i
                    self.execute_iteration(i, key)
        else:
            for i in iter_info[key]:
                self.sparams = self.update_iteration_params(key, i)
                self.execute_iteration(i, key)

    def execute_iteration(self, i, key):
        title, filename = self.update_title_filename(key)
        if self.filter_update:
            print_divider()
            print("FILTERING")
            print_divider()
            self.filter = Filter(
                load(DATASET_PKL),
                split=None,
                iou=True,
                horizon=True,
                theta=i if self.kwargs["theta"] else THETA,
                dist_interval=[i, i + (iter_info[key][1] - iter_info[key][0])] if self.kwargs["dist"] else [],
                visible_horizon=i if self.kwargs["visible"] else VISIBLE,
                iou_threshold=IOU_THRESHOLD + (i / 100) if self.kwargs["iou"] else IOU_THRESHOLD,
                claimed_distance_delta=i if self.kwargs["claimed"] else CLAIMED_DISTANCE,
                x_px_interval=[920 + (10 * i), 1000 - (10 * i)] if self.kwargs["x_px"] else X_PX_INTERVAL
            )

        protocol = self.initialize_protocol()
        summary = Summary(protocol)
        row = self.create_summary_row(list(self.sparams.values()), summary)
        self.df.loc[len(self.df)] = row
        self.table.append(row)
        self.df_to_csv(filename)
        self.print_table()

        label = self.generate_label(key, i)

    def update_iteration_params(self, key, i):
        param_map = {
            "threshold": "alpha",
            "scoring": "beta",
            "kernel": "k",
            "window": "w",
            "rate": "s",
        }
        self.sparams[param_map[key]] = i
        return self.sparams

    def update_title_filename(self, key):
        title_format = f"K={self.sparams['K']}_k={self.sparams['k']}_s={self.sparams['s']}"
        title = title_format.format(key=key)
        filename = f"{key}_sweep_{title}"
        return title, filename

    def generate_label(self, key, i):
        label = f"{label_info[key]} {round(i, 2)}"
        return label

    def create_summary_row(self, sparams, summary):
        row = sparams + summary.table(ri=(self.kwargs["threshold"] or self.kwargs["rate"]))
        return row

    def initialize_protocol(self):
        w = WINDOW_SIZE
        protocol = Protocol(
            self.filter.dataset,
            cap=self.sparams["Q"],
            encode=self.kwargs["encode"],
            dynamic=self.kwargs["dynamic"],
            show_sequence=self.kwargs["sequence"],
            sample_rate=self.sparams["s"],
            kernel_size=self.sparams["k"],
            encode_window=w,
            alpha=self.sparams["alpha"],
            beta=self.sparams["beta"],
            K=self.sparams["K"],
            squadron_size=SQUADRON,
            num_regions=self.sparams["r"],
            tau=self.sparams["t"],
        )
        return protocol

    def print_table(self):
        print(tabulate(self.table, headers='firstrow', tablefmt='fancy_grid'))

    def df_to_csv(self, filename):
        csv_path = os.path.join(f"plots/results/csv/{self.kwargs['save']}/{filename}.csv")
        self.df.to_csv(csv_path)


class Analysis:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def analysis(self):
        if self.kwargs["process"]:
            create_directory(f"plots/results/csv/{self.kwargs['save']}")
            create_directory(f"plots/results/csv/processed/")
        if self.kwargs["initialize"]:
            self.initialize_dataset()
        else:
            self.load_dataset()

        if self.kwargs["ab"]:
            self.initialize_ab_params()
        elif self.kwargs["threshold"]:
            self.initialize_ab_params()
            iter_info["scoring"] = [0.5] 
        sweep = Sweep(
            self.kwargs,
            filter_params=["dist", "theta", "visible", "iou", "claimed", "x_px"],
            sparams={
                'alpha': self.kwargs["alpha"], 'beta': self.kwargs["beta"],
                'K': int(self.kwargs["K"]), 'k': int(self.kwargs["k"]), 'Q': int(self.kwargs["Q"]),
                's': int(self.kwargs["s"]), 'r': int(self.kwargs["regions"]), 't': self.kwargs["tau"]
            },
            vc_colors=get_cmap("tab20").colors,
            adv_colors=get_cmap("tab20").colors
        )
        sweep.execute()
        if self.kwargs["process"]:
            if self.kwargs["ab"]:
                process_run(self.kwargs["save"])
            elif self.kwargs["threshold"]:
                process_run(self.kwargs["save"], threshold=True)

    def initialize_dataset(self):
        dataset = Dataset(verifier_label_directory, candidate_label_directory, sync=True, sensor=False)
        save(dataset, DATASET_PKL)

    def load_dataset(self):
        self.filter = load(FILTERED_DATASET_PKL)

    def initialize_ab_params(self):
        K = self.kwargs["K"]
        k = self.kwargs["k"]
        iter_info["scoring"] = [math.floor((x / K) * 100) / 100 for x in range(1, K)]
        iter_info["threshold"] = [math.floor((x / k) * 100) / 100 for x in range(1, k)]
