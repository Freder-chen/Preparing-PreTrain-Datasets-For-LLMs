import os
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


def draw_data(data_list, title, save_filename, start_x=None, end_x=None, count=True):
    start = start_x or min(data_list)
    end   = end_x   or max(data_list)
    
    if isinstance(start, str) and isinstance(end, str):
        counts = Counter(data_list)
        plt.bar(counts.keys(), counts.values(), edgecolor='black')
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    else:
        if count == True:
            interval = int((end - start) / 100) + 1
        else:
            interval = 0.01

        if start == end:
            start -= interval / 2
            end += interval / 2
        
        bins = np.arange(start, end + interval, interval)
        counts, _ = np.histogram(data_list, bins=bins)

        plt.bar(bins[:-1], counts, width=interval, edgecolor='black')

    
    plt.xlabel(f'{title}')
    plt.ylabel('Count')
    plt.title(f'{title} Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    plt.savefig(save_filename)
    plt.show()
    plt.clf()
