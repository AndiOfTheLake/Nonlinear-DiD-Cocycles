import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Plot contour plot


def plot_contour_kde(dt, col, ax, scatter=True):
    """
    Create side-by-side contour plots
    """
    if isinstance(dt, torch.Tensor):
        dt = dt.detach().cpu()
    df = pd.DataFrame(data=dt, columns=["x1", "x2"])
    sns.kdeplot(
        data=df,
        x="x1", y="x2", fill=True,
        cmap=col,
        alpha=0.5,
        ax=ax,
        bw_adjust=2  # Use the same parameter
    )
    if scatter: 
        ax.scatter(df["x1"], df["x2"],
                   s=8, c="k", alpha=0.45, linewidths=0.2, edgecolors="white",
                   zorder=3)  # draw above the filled KDE
        

def contour_quick(dt, col="Blues"):
    """
    Plot a simple contour plot for a quick view of the distribution; 
    no additional arguments passed
    """

    if isinstance(dt, torch.Tensor):
        dt = dt.detach().cpu()
    df = pd.DataFrame(data=dt, columns=["x1", "x2"])
    sns.kdeplot(
        data=df,
        x="x1", y="x2", fill=True,
        cmap=col,
        alpha=0.5,
        bw_adjust=2  # Use the same parameter
    )

# Update graphics for each subplot (write a function to avoid repetition)


def set_contour(axes, i, title):
    axes[i].set_xlabel(None)
    axes[i].set_ylabel(None)
    axes[i].set_title(title)

# Automatically set xlim and ylim for all subplots based on input data
# https://stackoverflow.com/questions/31006971/setting-the-same-axis-limits-for-all-subplots
def set_xylim(lst_dt):   
    dt_combined = torch.cat(lst_dt, dim=-2)
    def get_lim(dt, i):
        xi_min = dt[:, i].min()
        xi_max = dt[:, i].max()
        xi_sd = dt[:, i].std()
        return tuple(torch.tensor([xi_min, xi_max]) + torch.tensor([-1, 1]) * xi_sd)
    return([get_lim(dt_combined, 0), 
            get_lim(dt_combined, 1)])

# Plot the losses against the number of epochs


def plot_loss(loss, title):
    plt.figure()
    plt.plot(loss)
    plt.axvline(x=loss.argmin(), color="red")
    plt.text(loss.argmin(),
             # y coordinate for the text
             plt.gca().get_yticks().mean(),
             f'epoch {loss.argmin()}',
             va="center",
             ha="right",
             color="red")
    plt.xlabel("Epoch")
    plt.title(title)

