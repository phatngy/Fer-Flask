import matplotlib.pyplot as plt
import glob
import os
import numpy as np


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct, absolute)

def pie_plot(classes, counts):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(counts, autopct=lambda pct: func(pct, counts),
                                        textprops=dict(color="w"))    
    ax.legend(wedges, [f'{k} ({v})' for k,v  in zip(classes, counts)],
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))    
    plt.setp(autotexts, size=6, weight="bold")
    return fig, ax
def stats(in_dir):
    classes = []
    counts = []
    for label in glob.glob(os.path.join(in_dir, '*')):
        label_name = label.split('/')[-1]
        # classes.append(label_name)
        count = os.listdir(label)
        # counts.append(count)
        print(label_name, len(count))
    # return classes, counts

stats('./FER/output/Training')