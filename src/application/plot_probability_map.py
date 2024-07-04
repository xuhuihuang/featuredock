import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

## convert percentile to transparency value
def get_transparency(d):
    if d >= 0 and d < 0.5:
        return 0.5
    elif d >= 0.5 and d < 0.75:
        return 0.3
    else:
        return 0


def plot_probability(probfile, voxelfile, cutoff, colormap="Blues", relative=True, is_rank=False, plot_every=1):
    with open(probfile, 'rb') as file:
        probs = pickle.load(file)
        probs = probs[:, 1]
    with open(voxelfile, 'rb') as file:
        coords = pickle.load(file)
    
    ## select grids of interest
    cutoff = cutoff
    mask = probs>=cutoff
    sele_coords = coords[mask][::plot_every]
    sele_probs = probs[mask][::plot_every]
    vmin = np.min(sele_probs)
    vmax = np.max(sele_probs)
    if relative:
        rescaled_sele_probs = (sele_probs-vmin)/(vmax-vmin)
    else:
        rescaled_sele_probs = sele_probs
    # cm = mpl.colormaps[colormap]
    cm = plt.cm.get_cmap(colormap)
    newcmp = ListedColormap(cm(rescaled_sele_probs))
    newcmp = newcmp.colors[:, :3]

    if is_rank:
        rescaled_sort = np.argsort(rescaled_sele_probs)
        length = rescaled_sort.shape[0]
        rescaled_percentile = {idx: rank/length for rank, idx in enumerate(rescaled_sort)}
    else:
        rescaled_percentile = {idx: rescaled_p for idx, rescaled_p in enumerate(rescaled_sele_probs)}
    ## plot
    cmd.set("sphere_scale", 0.1)
    cmd.set("transparency_mode", 1)
    # cmd.set_color("color", color)
    name = f"probabilities"
    cmd.delete(f"{name}")
    cmd.create(f"{name}", 'none')

    for i in range(sele_coords.shape[0]):
        color = newcmp[i]
        cmd.pseudoatom(f"{name}", pos=list(sele_coords[i]), name=f"{name}_{i}")
        cmd.set_color(f'{name}_color{i}', list(color))
        cmd.color(f'{name}_color{i}', f'{name} and name {name}_{i}')
        cmd.set('transparency', get_transparency(rescaled_percentile[i]), f"{name} and name {name}_{i}")
    cmd.show("surface", f"{name}")


# plot_probability(probfile, voxelfile, cutoff=0.8, colormap="Blues", relative=True, is_rank=False, plot_every=1)