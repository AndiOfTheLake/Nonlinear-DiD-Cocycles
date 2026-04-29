
# Cocycle applied to continuous treatments
import argparse
import torch
import numpy as np
import math
import zuko
import inspect
import os
import re
from pathlib import Path
import random
from torch.utils.data import DataLoader, TensorDataset
# Load source functions
from source.kernels_new import median_heuristic  # Written by Hugh Dance
from source.miscv00 import *
from source.cmmdulossv00 import *
from source.funsplotv00 import *

# ---- Housekeeping ----

parser = argparse.ArgumentParser()
parser.add_argument("--effect_type", type=str, required=True)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--script_ver", type=int, required=True)
args = parser.parse_args()


scheme = "B"
effect_type = args.effect_type
n_arm = args.n
version = args.script_ver

# n_arm = 1000
# effect_type = "id"
# scheme = "B"
# version = 0

n_ctrl = n_arm
n_trted = n_arm


# True natural trend map
d = 2  # Outcome dimension, i.e., R^d
d_cond_var = 2  # dimension of the conditioning variable

# True natural trend
torch.manual_seed(0)
p = zuko.flows.autoregressive.MAF(features=d, transforms=5,
                                  hidden_features=(64, 64,))

# Mean and covariance for the control group
loc = torch.tensor([0., 0]) 
cov = torch.tensor([3., 0., 0., 3]).reshape(-1, 2)
noise = "indpt"


f = give_f(effect_type)


def R(t): 
    """
    Returns a 2D rotation matrix parameterized by t
    """
    pi = torch.tensor(math.pi)
    a = torch.cos(2 * pi * t)
    b = torch.sin(2 * pi * t)
    return torch.tensor([a, -b, b, a]).reshape(2, 2)

def cov_trt(t): 
    """
    Returns a 2D covariance matrix for the treated at dose t
    """
    a = torch.tensor(-2.9) * t + torch.tensor(3.)
    b = torch.tensor(-0.05) * t
    return torch.tensor([a, b, b, a]).reshape(2, 2)


# Create project name; will be used for plots as well

# proj = ""
proj = rf"cont-{scheme}-{effect_type}-{n_arm:05d}-v{version:02d}"
proj

# ---- Define device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ---- create directory for plots ----
# Place all plots in one place
dir_plots = "all-cont-" + scheme + "-plots" + rf'-v{version:02d}'
dir_plots
os.makedirs(dir_plots, exist_ok=True)


# ## [Section]: Generate data


# ----- data -----

def draw_samples(seed, 
                 loc=loc, cov=cov, d=d, 
                 n_ctrl=n_ctrl, n_trted=n_trted, 
                 R=R, # Rotation matrix
                 cov_trt=cov_trt, 
                 p=p, # natural trend map
                 effect_type=effect_type,
                 noise="indpt"): 
    """Draws samples for Experiment 2.

    Parameters
    ----------
    seed : int
        Random seed. 
    loc : torch.Tensor
        The mean of the outcome distribution at dose = 0 and time = 0.
    cov : torch.Tensor
        The covariance of the outcome distribution at dose = 0 and time = 0.
    d : int
        The outcome dimension.
    n_ctrl : int
        Size of the control group.
    n_trted : int
        Size of the treated group.
    R : function
        A rotation matrix as a function of dose.
    p : zuko.flows.MAF
        The (nonlinear) natural trend map. 
    effect_type : str
        Type of dose response curve (id or square root).
    noise : str
        Noise type (independent noise draws or fixed noise draws), by default "indpt". 

    Returns
    -------
    list
        A list of 4 tensors corresponding to the controls and the treated at times 0 and 1.
        Each tensor includes the outcomes and the conditioning variables. 

    Raises
    ------
    ValueError
        Restriction on noise types. 
    ValueError
        Restriction on effect types. 
    """
    if noise not in ["indpt", "fixed"]:
        raise ValueError('noise must be either "indpt" or "fixed"')
    if effect_type not in ["id", "sqrt"]:
        raise ValueError('effect_type must be "id" or "sqrt".')   
    # Forces the user to supply a seed
    torch.manual_seed(seed)
    # Control samples
    mvt = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    dt_00 = torch.cat(
        [mvt.sample((n_ctrl, )), 
         torch.tensor([0., 0]).unsqueeze(-2).expand(n_ctrl, -1)], 
        dim=-1
    )
    tmp01 = dt_00[:, :d] if noise == "fixed" else mvt.sample((n_ctrl, ))
    
    dt_01 = torch.cat(
        [p().transform(tmp01),
        torch.tensor([0., 1]).unsqueeze(-2).expand(n_ctrl, -1)],
        dim=-1
    )
    t = torch.rand(n_trted).clamp_min(1e-6).clamp_max(1 - 1e-6)
    
    # Three things: shift,  rotation, and covariance
    loc_new = t.unsqueeze(-1) + loc # Create new mean vectors via broadcasting    

    def get_treated_outcome(t): 
        dt = torch.empty([n_trted, d])
        for i in range(n_trted):
            mvt_trt = torch.distributions.MultivariateNormal(
                loc=loc_new[i, :],
                covariance_matrix=R(t[i]) @ cov_trt(t[i]) @ R(t[i]).t()
            )
            dt[i, :] = mvt_trt.sample()
        return dt
    dt_10_outcome = get_treated_outcome(t)
    tmp02 = dt_10_outcome if noise == "fixed" else get_treated_outcome(t)
    
    true_counterfactual = p().transform(tmp02)    
    dt_11_outcome = true_counterfactual\
        + (t if effect_type == "id" else torch.sqrt(t))\
            .unsqueeze(-1)
    dt_10 = torch.cat(
        [dt_10_outcome, t.unsqueeze(-1), torch.zeros([n_trted, 1])], dim=-1
    )
    dt_11 = torch.cat(
        [dt_11_outcome, t.unsqueeze(-1), torch.ones([n_trted, 1])], dim=-1
    )
 
    return [dt.detach() for dt in [dt_00, dt_01, dt_10, dt_11]]



def prepare_train_test_data(lst_dt, prop_validate=0.2, 
                            d=d, d_cond_var=d_cond_var):    
    """Prepares training and test data for Expriment 2. 

    Parameters
    ----------
    lst_dt : list
        A list of tensors. 
        This is expected to be the output of `draw_samples()`. 
    prop_validate : float
        The proportion of data designated to be in the test set, by default 0.2.
    d : int
        The outcome dimension.
    d_cond_var : int
        Dimension of the conditioning variable. 

    Returns
    -------
    dict
        A dictionary containing training and validation sets. 
        Some are lists while others are tensors. 
        Some are standardized while others are on the original scale. 
        Look inside the function for details. 
    """
    # All observations; a tensor; original scale
    dt_all_orig = torch.cat(lst_dt, dim=-2) 
    # All observations; a tensor; standardized to zero mean and unit variance
    dt_all_std = torch.cat([standardize_sample(dt_all_orig[:, :d]), 
                            dt_all_orig[:, -d_cond_var:]], dim=-1)
    # Location and scale parameters of the affine transformation/standardization
    rst_std = standardize_sample(dt_all_orig[:, :d], keep=True)
    # Reconstruct the transformation 
    def std(dt, rst_std=rst_std):   
        outcome = (rst_std['scale'].inverse() @ (dt[:, :d] - rst_std['loc'])\
                   .unsqueeze(-1))\
                    .squeeze(-1)
        return torch.cat([outcome, dt[:, -d_cond_var:]], dim=-1)
    
    # torch.allclose(std(dt_all_orig), dt_all_std)

    # Observations in the training/validation set; a list; original scale
    lst_train_orig = [dt[:-int(dt.shape[0] * prop_validate), :] for dt in lst_dt]
    lst_test_orig = [dt[-int(dt.shape[0] * prop_validate):, :]
                     for dt in lst_dt]

    # Observations in the training/validation set; a list; standardized
    lst_train_std = [std(dt) for dt in lst_train_orig]
    lst_test_std = [std(dt) for dt in lst_test_orig]

    # Observations in the training/validation set; a tensor; original scale
    dt_train_orig = torch.cat(lst_train_orig, dim=-2)
    dt_test_orig = torch.cat(lst_test_orig, dim=-2)

    # Observations in the traing/validation set; a tensor; standardized
    dt_train_std = torch.cat(lst_train_std, dim=-2)
    dt_test_std = torch.cat(lst_test_std, dim=-2)

    return {
        'dt_all_orig': dt_all_orig, 
        'dt_all_std': dt_all_std, 
        'lst_train_orig': lst_train_orig, 
        'lst_test_orig': lst_test_orig, 
        'lst_train_std': lst_train_std,
        'lst_test_std': lst_test_std,
        'dt_train_orig': dt_train_orig, 
        'dt_test_orig': dt_test_orig, 
        'dt_train_std': dt_train_std, 
        'dt_test_std': dt_test_std
    }



# Draw samples on the original scale
lst_dt = draw_samples(seed=0, noise="indpt", n_ctrl=n_ctrl, n_trted=n_trted)
# Create a dictionary of original and standardized samples
dict_dt = prepare_train_test_data(lst_dt)

sample_std = dict_dt['dt_all_std']
for i in (range(d)):
    print(
        f"y_{i} standardized: min = {sample_std[:, i].min()}; max = {sample_std[:, i].max()}")

loc_scale = standardize_sample(dict_dt['dt_all_orig'][:, :d], keep=True)

sample_std[:, :d].mean(dim=-2)  # Should be very close to zero
torch.cov(sample_std[:, :d].T)  # Should be very close to the identity matrix



## [Section]: Plot standardized training data

lst_train_std = dict_dt['lst_train_std']

sns.set_theme(style="white")
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
contour_xlim, contour_ylim = (-5, 5), (-5, 5)
plt.setp(axes, xlim=contour_xlim, ylim=contour_ylim)

for ax, dt, title in zip(axes.flat, lst_train_std, 
                         [r"$s_{(0, 0)}$", r"$s_{(0, 1)}$",
                          r"$s_{(\tau, 0)}$",r"$s_{(\tau, 1)}$"]): 
    plot_contour_kde(dt=dt[:, :d], col="Blues", ax=ax, scatter=False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(title)

fig.set_size_inches((15, 10))
plt.savefig(rf'{dir_plots}/{proj}-00samples.png', bbox_inches="tight")
plt.show()



# ## [Section]: Write training loop


# Define a SINGLE conditional flow
torch.manual_seed(0)
flows = zuko.flows.NSF(features=d, context=d_cond_var,
                       transforms=5, bins=8, 
                       hidden_features=(128, 128,))
flows = flows.to(device)

count_param(flows) # Total number of parameters

opt = torch.optim.Adam(flows.parameters(), lr=0.001)


# ---- Median heuristic
print(inspect.getsource(median_heuristic))  # Hugh's implementation
# Set bandwidth for Gaussian kernerl;
# We subsample to avoid memory overload (still takes a while)

torch.manual_seed(0)  # Reproducible subsampling
bandwidth = median_heuristic(X=dict_dt['dt_train_std'][:, :d]) \
    if dict_dt['dt_train_std'].shape[0] <= 6000 \
    else median_heuristic(X=subsample(dict_dt['dt_train_std'])[:, :d])

bandwidth = bandwidth.to(device)
print(bandwidth)


# ## [Section]: Set up checkpoints


# -------------------- config --------------------
EPOCHS = 1000
CKPT_DIR = "ckpts" + proj
CKPT_STRIDE = 10           # save a full checkpoint every K epochs
BASE_SEED = 123
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 800

os.makedirs(CKPT_DIR, exist_ok=True)

# -------------------- seeding --------------------


def seed_all(s: int):
    """Sets random seed for PyTorch, NumPy, and Python's random module simultaneously.

    Parameters
    ----------
    s : int
        Seed value. 
    """
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def begin_epoch_seed(epoch: int):
    """Sets seed for a particular epoch 
    to keep per-epoch randomness deterministic.

    Parameters
    ----------
    epoch : int
        Epoch number. 
    """    
    s = BASE_SEED + epoch
    seed_all(s)


seed_all(BASE_SEED)  # initial/global seed

# -------------------- data loaders --------------------


def make_train_loader(dataset, epoch):
    """Creates a DataLoader for the training set at an epoch.
    It uses a deterministic per-epoch shuffling. 


    Parameters
    ----------
    dataset : torch.utils.data.TensorDataset
        Input data set. 
    epoch : int
        Epoch number. 

    Returns
    -------
    torch.utils.data.DataLoader
        Data loader for training set. 
    """
    # Deterministic shuffling via per-epoch generator
    g = torch.Generator()
    g.manual_seed(BASE_SEED + epoch)
    return DataLoader(dataset, batch_size=BATCH_SIZE_TRAIN, 
                      shuffle=True, generator=g)


def make_test_loader(dataset, epoch):
    """Creates a DataLoader for the test set at an epoch.
    There is no shuffling. 


    Parameters
    ----------
    dataset : torch.utils.data.TensorDataset
        Input data set. 
    epoch : int
        Epoch number. 

    Returns
    -------
    torch.utils.data.DataLoader
        Data loader for test set. 
    """
    g = torch.Generator() # redundant
    g.manual_seed(BASE_SEED + epoch) # redundant
    return DataLoader(dataset, batch_size=BATCH_SIZE_TEST, 
                      shuffle=False)


# ---- True loss
torch.manual_seed(0)

# ---- Training loop
torch.manual_seed(1)  # To ensure reproducibility
scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=10 ** 9, gamma=1.0)


# -------------------- train / eval --------------------


def train_one_epoch(epoch, epochs, flows, opt, loader, bandwidth):
    """Trains the model for one epoch and 
    computes the per-epoch *training* CMMD-U loss averaged over minibatches. 

    Parameters
    ----------
    epoch : int
        Epoch number. 
    epochs : int
        Total number of epochs. 
    flows : zuko.flows.NSF
        Neural spline flow to be trained to model the cocycle.
    opt : torch.optim.Adam
        ADAM optimizer. 
    loader : torch.utils.data.DataLoader
        Loader for training data. 
    bandwidth : torch.Tensor
        Bandwidth for the kernel (usually calculated using the median heuristic).

    Returns
    -------
    float
        Per-epoch *training* CMMD-U loss averaged over minibatches.
    """
    cmmd_u_loss = 0.
    for y_batch, cond_var_batch in loader:
        loss = compute_cmmdu(y_batch, cond_var_batch, bandwidth, flows)
        opt.zero_grad()
        loss.backward()
        opt.step()
        cmmd_u_loss += loss.item()

    if epoch in (0, epochs - 1) or (epoch % 100 == 0):
        print(
            f"Epoch {epoch} training_loss: {cmmd_u_loss / len(loader):.4f}"
        )
    return cmmd_u_loss / len(loader)

# # Some checking; works
# train_one_epoch(1, 10, flows, opt, loader, bandwidth)

@torch.no_grad()
def test_one_epoch(epoch, epochs, flows, loader, bandwidth):
    """Computes the per-epoch *validation* CMMD-U loss averaged over minibatches. 

    Parameters
    ----------
    epoch : int
        Epoch number. 
    epochs : int
        Total number of epochs. 
    flows : zuko.flows.NSF
        Neural spline flow to be trained to model the cocycle.    
    loader : torch.utils.data.DataLoader
        Loader for training data. 
    bandwidth : torch.Tensor
        Bandwidth for the kernel (usually calculated using the median heuristic).

    Returns
    -------
    float
        Per-epoch *validation* CMMD-U loss averaged over minibatches.
    """
    cmmd_u_loss = 0.
    for y_batch, cond_var_batch in loader:
        loss = compute_cmmdu(y_batch, cond_var_batch, bandwidth, flows)        
        cmmd_u_loss += loss.item()

    if epoch in (0, epochs - 1) or (epoch % 100 == 0):
        print(
            f"Epoch {epoch} validation_loss: {cmmd_u_loss / len(loader):.4f}"
        )
    return cmmd_u_loss / len(loader)



def create_tensordata(lst_outcome, lst_cond_var, device=device):
    """Creates training/validation data sets and put them in the correct device.

    Parameters
    ----------
    lst_outcome : list
        List of standardized outcome tensors. 
    lst_cond_var : list
        List of tensors of the corresponding conditioning variables. 
    device : torch.device
        The device where the `TensorDataset` is stored. 

    Returns
    -------
    torch.utils.data.TensorDataset
        The `TensorDataset` to be provided to `train_all()` and subsequently 
        to `make_train_loader()` or `make_test_loader()`.
    """
    return TensorDataset(
        torch.cat(lst_outcome, dim=-2).to(device).float(),
        torch.cat(lst_cond_var, dim=-2).to(device).float()
    )




# -------------------- checkpoint I/O --------------------


def ckpt_path(epoch):
    """Creates a str that indicates the path to a checkpoint of the model 
    state after the training at `epoch` is completed. 

    Parameters
    ----------
    epoch : int
        Epoch number.

    Returns
    -------
    str
        Path to checkpoint.
    """
    return f"{CKPT_DIR}/full_{epoch:04d}.pt"


def save_full_ckpt(epoch):
    """Saves a checkpoint for the model state 
    after the training at `epoch` is completed. 
    The location of the saved checkpoint is specified by a function call of `ckpt_path()`.
    The checkpoint is a dictionary with the following keys: 
        - "epoch": int; the epoch number.
        - "flows": collections.OrderedDict; the state dictionary of `flows`.
        - "optimizer": collections.OrderedDict; the state dictionary of `opt`.
        - "scheduler": collections.OrderedDict; the state dictionary of `scheduler`.

    Parameters    
    ----------
    epoch : int
        Epoch number.
    """
    torch.save({
        "epoch": epoch,
        "flows": flows.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_path(epoch))


def load_full_ckpt(epoch, device=device):
    """Loads a checkpoint after training is complete for epoch `epoch` 
    to the device specified by `device`. 

    Parameters
    ----------
    epoch : int
        Epoch number. 
    device : torch.device        

    Returns
    -------
    float
        Returns the epoch number.
    """
    ckpt = torch.load(ckpt_path(epoch), map_location=device)
    flows.load_state_dict(ckpt["flows"])
    opt.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"]


# -------------------- workflows --------------------

# We must use the standardized data here
lst_test_std = dict_dt['lst_test_std']

lst_outcome_train = [dt[:, :d] for dt in lst_train_std]
lst_cond_var_train = [dt[:, -d_cond_var:] for dt in lst_train_std]
lst_outcome_test = [dt[:, :d] for dt in lst_test_std]
lst_cond_var_test = [dt[:, -d_cond_var:] for dt in lst_test_std]

dt_train = create_tensordata(lst_outcome_train, lst_cond_var_train)
dt_test = create_tensordata(lst_outcome_test, lst_cond_var_test)


def train_all(flows=flows,
              opt=opt,
              bandwidth=bandwidth,
              epochs=EPOCHS,
              dt_train=dt_train,
              dt_test=dt_test,
              ckpt_stride=CKPT_STRIDE):
    """Trains the cocycle for all epochs. 
    Computes the training and validation losses after each epoch.
    Saves checkpoints. 
    Returns losses.

    Parameters
    ----------
    flows : zuko.flows.NSF
        Neural spline flow to be trained to model the cocycle.
    opt : torch.optim.Adam
        ADAM optimizer. 
    bandwidth : torch.Tensor
        Bandwidth for the kernel (usually calculated using the median heuristic).
    epochs : int
        Total number of epochs. 
    dt_train : torch.utils.data.TensorDataset
        The `TensorDataset` passed to `make_train_loader()`.
    dt_test : torch.utils.data.TensorDataset
        The `TensorDataset` passed to `make_test_loader()`.
    ckpt_stride : int
        The "stride" of checkpoints 
        (i.e., a checkpoint is created every `ckpt_stride` epochs).

    Returns
    -------
    loss_training : torch.Tensor
        A tensor of training losses of shape (epochs,).
    loss_validation : torch.Tensor
        A tensor of validation losses of shape (epochs,).
    """
    
    loss_training = torch.empty(epochs)
    loss_validation = torch.empty(epochs)   

    for epoch in range(epochs):
        begin_epoch_seed(epoch)
        # Training
        train_loader = make_train_loader(dataset=dt_train, epoch=epoch)
        loss_training[epoch] = train_one_epoch(
            epoch=epoch, epochs=epochs, flows=flows, opt=opt,
            loader=train_loader, bandwidth=bandwidth)
        scheduler.step()
        # Validating
        test_loader = make_test_loader(dataset=dt_test, epoch=epoch)
        loss_validation[epoch] = test_one_epoch(
            epoch=epoch, epochs=epochs, flows=flows, 
            loader=test_loader, bandwidth=bandwidth
        )

        if (epoch % ckpt_stride == 0) or epoch == (epochs - 1):
            save_full_ckpt(epoch)

    return loss_training, loss_validation


print(EPOCHS)


## [Section]: Execute training loop


dir_saved_vars = "proj-vars" + proj  # directory of saved variables
os.makedirs(dir_saved_vars, exist_ok=True)

# Check if the losses are already saved; if so, no need to rerun
print("losses.pt" in os.listdir(str(Path(dir_saved_vars))))


if "losses.pt" in os.listdir(str(Path(dir_saved_vars))):
    losses = torch.load(str(Path(dir_saved_vars) / "losses.pt"))
    loss_training, loss_validation = losses["loss_training"], losses["loss_validation"]
else:
    loss_training, loss_validation = train_all(epochs=EPOCHS)

    # Training is time-comsuming so we save training and validation losses
    # once the training loop is complete
    # Next time the script is run it will automatically load the
    # saved losses if they are available

    # This saves those variables we want to save as a dictionary
    # * means anything after this has to be passed as keyword argument
    def save_vars(names: list, *, scope):
        return {name: scope[name] for name in names}

    losses = save_vars(
        names=["loss_training", "loss_validation"], scope=globals())
    path_losses = str(Path(dir_saved_vars) / "losses.pt")
    torch.save(losses, path_losses)


# Note that loss_training and loss_validation are on the CPU, NOT EVER on the GPU
[x.device for x in [loss_training, loss_validation]]


# ## [Section]: Plot losses


# Plot the losses against the number of epochs

plot_loss(loss_training, "Training loss")
plt.savefig(rf'{dir_plots}/{proj}-01train.png', bbox_inches="tight")
plt.show()

plot_loss(loss_validation, "Validation loss")
plt.savefig(rf'{dir_plots}/{proj}-02validation.png', bbox_inches="tight")
plt.show()

loss_training.argmin()
loss_validation.argmin()
loss_training.min()
loss_validation.min()


# ## [Section]: Recover the flows based on checkpoints


def recover_flows(n,
                  flows=flows,
                  opt=opt,
                  bandwidth=bandwidth,
                  dt_train=dt_train,
                  epochs=EPOCHS,
                  ckpt_stride=CKPT_STRIDE):
    """Reconstructs and saves model state at epoch `n`. 
    It first checks if the checkpoint for the requested epoch is already available. 
    If so, it returns the model state using that checkpoint. 
    Otherwise, it retrains the model from checkpoint saved after the most recent epoch
    before epoch `n` and saves and new checkpoint once the training is completed. 
    
    Parameters
    ----------
    n : torch.Tensor | int
        Epoch number `n`.
    flows : zuko.flows.NSF
        Neural spline flow to be trained to model the cocycle.
    opt : torch.optim.Adam
        ADAM optimizer. 
    bandwidth : torch.Tensor
        Bandwidth for the kernel (usually calculated using the median heuristic).
    dt_train : torch.utils.data.TensorDataset
        The `TensorDataset` passed to `make_train_loader()`.
    epochs : int
        Total number of epochs. 
    ckpt_stride : int
        The "stride" of checkpoints 
        (i.e., a checkpoint is created every `ckpt_stride` epochs).
    """
    if torch.is_tensor(n):
        n = int(n.item())
    else:
        n = int(n)

    # Fast path: exact checkpoint exists
    s = int(np.floor(n / ckpt_stride) * ckpt_stride)
    pattern = re.compile(rf"full_{n:04d}\.pt")
    checkpoints = os.listdir(CKPT_DIR)
    if n == s or n == (epochs - 1) or any(pattern.match(x) for x in checkpoints):
        load_full_ckpt(n)
        print(
            f"Full checkpoint already saved for epoch {n} (no replay) -> {ckpt_path(n)}")
        return

    # Load s and replay to n
    load_full_ckpt(s)
    for epoch in range(s + 1, n + 1):
        begin_epoch_seed(epoch)
        train_loader = make_train_loader(dataset=dt_train, epoch=epoch)
        train_one_epoch(
            epoch=epoch, epochs=epochs, flows=flows, opt=opt,
            loader=train_loader, bandwidth=bandwidth)
        scheduler.step()

    save_full_ckpt(n)
    print(
        f"Saved checkpoint for epoch {n} (replayed {n - s} epoch(s))")


recover_flows(n=loss_validation.argmin())


# ## [Section]: Plot cocycles

def sort_by_treatment(dt): 
    """Sorts data by increasing treatment value"""
    indeces = dt[:, -d_cond_var:][:, 0].argsort()
    return dt[indeces, :]


def cocycle(y, 
          c_to=torch.tensor([0., 1]), 
          c_from=torch.tensor([0., 0]),
          flows=flows, loc_scale=loc_scale,
          device=device):
    """Learned TMI transport maps between outcomes on the *original* scale 
    indexed by conditioning variables.

    Parameters
    ----------
    y : torch.Tensor
        Outcome tensor of shape `(n, d)` from the source distribution/distributions 
        on the *original* scale.
    c_to : torch.Tensor
        Index/indeces of the target distribution/distributions. 
        Each row of `c_to` is a conditioning variable that indexes a target distribution.
    c_from : torch.Tensor
        Index/indeces of the source distribution/distributions.
        Each row of `c_from" is a conditioning variable that indexes a source distribution.
    flows : zuko.flows.NSF
        Neural spline flow to be trained to model the cocycle.
    loc_scale : dict[str, torch.Tensor]
        A dictionary including the location and scale parameters 
        for a standardizing transform from the original scale. 
        This is expected to be the output of `standardize_sample()` with `keep=True`. 
    device : torch.device

    Returns
    -------
    torch.Tensor
        Outcome tensor of shape `(n, d)` from the target distribution/distributions 
        on the *original* scale.
    """
    y = y.to(device)
    c_to = c_to.to(device)
    c_from = c_from.to(device)

    # Affine transformation; standardization and its inverse
    def affine(y, l_s, inv=False, device=device):
        l_s = {k: v.to(device) for k, v in l_s.items()}
        if inv:
            return (l_s['scale'] @ y.unsqueeze(-1)).squeeze(-1) + l_s['loc']
        else:
            return (l_s['scale'].inverse() @ (y - l_s['loc']).unsqueeze(-1)).squeeze(-1)
    # TMI transport maps on *standardized samples*
    def tmi(c_to, c_from, y_from, flows=flows):              
       with torch.no_grad():
            return flows(c=c_to).transform.inv(
                flows(c=c_from).transform(y_from)
            )
    return affine(
        y=tmi(c_to, c_from,
                  affine(y, l_s=loc_scale), flows=flows),
        l_s=loc_scale,
        inv=True
    )

# Estimated natural trend function
def trend(y, flows=flows, loc_scale=loc_scale,
          device=device): 
    """Learned natural trend map on the *original* scale. 

    Parameters
    ----------
    y : torch.Tensor
        Outcome tensor of shape `(n, d)` on the *original* scale 
        at time = 0 (i.e., pre-treatment outcomes).
    flows : zuko.flows.NSF
        Neural spline flow to be trained to model the cocycle.
    loc_scale : dict[str, torch.Tensor]
        A dictionary including the location and scale parameters 
        for a standardizing transform from the original scale. 
        This is expected to be the output of `standardize_sample()` with `keep=True`. 
    device : torch.device

    Returns
    -------
    torch.Tensor
        Outcome tensor of shape `(n, d)` on the *original* scale 
        at time = 1 (i.e., post-treatment outcomes).
    """
    return cocycle(y,
                   c_to=torch.tensor([0., 1]),
                   c_from=torch.tensor([0., 0]),
                   flows=flows, loc_scale=loc_scale,
                   device=device)



def estimate_mean_effects(lst_dt=lst_dt, n_trted=n_trted, d=d, device=device, 
                         dir_saved_vars=dir_saved_vars, 
                         flows=flows,
                         fast=False,
                         q_lower = 0.25, q_upper = 0.75, 
                         filename_mean_effects="mean_effects", 
                         filename_q_lower="quantile_lower", 
                         filename_q_upper="quantile_upper"):
    """Computes the plug-in estimate of the ATT at each dose level 
    in the observed data. At each coordinate of the estimated ATT, the sample 
    `q_lower`th and `q_upper`th percentiles are also computed. 

    Parameters
    ----------
    lst_dt : list
        A complete list of observed outcome tensors on the *original* scale. 
        This is expected to be the output of `draw_samples()`. 
    n_trted : int
        Size of the treated group.
    d : int
        The outcome dimension.
    device : torch.device
    dir_saved_vars : str
        Directory of saved variables.
    flows : zuko.flows.NSF
        Neural spline flow to be trained to model the cocycle.
    fast : bool
        Whether to use a fast path by checking 
        if the quantities have already been computed, by default False.
    q_lower : float
        The lower quantile of the estimated ATT, by default 0.25.
    q_upper : float
        The upper quantile of the estimated ATT, by default 0.75.
    filename_mean_effects : str
        File name for the tensor of mean effects. 
    filename_q_lower : str
        File name for the tensor of lower quantiles.  
    filename_q_upper : str
        File name for the tensor of upper quantiles. 

    Returns
    -------
    rst_dict : dict[str, torch.Tensor]
        A dictionary containing three tensors:
        mean effects, lower quantiles, and upper quantiles. 
    
    Notes
    -----
    These three tensors are also saved when the function is called for the first time.
    """
    filenames = [filename_mean_effects, filename_q_lower, filename_q_upper]
    filenames_ext = [file + ".pt" for file in filenames]
    saved_files = os.listdir(str(Path(dir_saved_vars)))    
    rst_dict = {}
    if fast and all(file in saved_files for file in filenames_ext):
        print(rf'Tensors loaded from {str(Path(dir_saved_vars))}')
        for i in range(len(filenames)):
            rst_dict[filenames[i]] =\
                  torch.load(str(Path(dir_saved_vars) / rf'{filenames[i]}.pt'),
                                                map_location=device)        
        return rst_dict
    
    else: 
        mean_effects = torch.empty([n_trted, d], device=device)
        quantile_lower = torch.empty([n_trted, d], device=device)
        quantile_upper = torch.empty([n_trted, d], device=device)
        # Sorted data
        [sorted_00, sorted_01, sorted_10, sorted_11] = \
            [sort_by_treatment(dt).to(device) for dt in lst_dt]
        
        dt_to_be_sent0 = torch.cat([sorted_10[:, :], sorted_00[:, :]], dim=-2)
        dt_to_be_sent1 = torch.cat([sorted_11[:, :], sorted_01[:, :]], dim=-2)

        def move_data(dt_to_be_sent, df_trted, i):             
            tmp = cocycle(c_to=df_trted[i, -d_cond_var:],                         
                          c_from=dt_to_be_sent[:, -d_cond_var:],
                          y=dt_to_be_sent[:, :d],
                          flows=flows)
            return tmp        

        for i in range(n_trted):
            with torch.no_grad():
                tmp0 = move_data(dt_to_be_sent=dt_to_be_sent0, 
                                 i=i, 
                                 df_trted=sorted_10)
                # Esimated counterfactual distribution
                est_counterfactual = trend(tmp0)
                # Estimated observed distribution 
                est_observed = move_data(
                    dt_to_be_sent=dt_to_be_sent1, 
                    i=i, 
                    df_trted=sorted_11
                )
                diff = est_observed - est_counterfactual

            quantile_lower[i, :] = torch.quantile(diff, q_lower, dim=0)
            quantile_upper[i, :] = torch.quantile(diff, q_upper, dim=0)                
            mean_effects[i, :] = diff.mean(dim=-2)

        to_be_saved = [mean_effects, quantile_lower, quantile_upper]
        for i in range(len(filenames)):
            rst_dict[filenames[i]] = to_be_saved[i]
            torch.save(to_be_saved[i],
                       str(Path(dir_saved_vars) / rf'{filenames[i]}.pt'))
        print(
            rf'Estimated mean effects saved under {str(Path(dir_saved_vars))}.'
            )
    return rst_dict


dict_mean_effects = estimate_mean_effects(fast=False)
mean_eff_est = dict_mean_effects['mean_effects']
ql = dict_mean_effects['quantile_lower']
qu = dict_mean_effects['quantile_upper']



# Sorted data
[sorted_00, sorted_01, sorted_10, sorted_11] = [sort_by_treatment(dt).to(device)
                                                for dt in lst_dt]
# Sniff check; should be true
torch.allclose(sorted_11[:, -d_cond_var:][:, 0], 
               sorted_10[:, -d_cond_var:][:, 0])

t = sorted_11[:, -d_cond_var:][:, 0]

mean_eff_true = f(t).unsqueeze(-1).expand(len(t), 2)

def plot_effect(mean_eff_est=mean_eff_est, 
                mean_eff_true=mean_eff_true,
                ql=ql, qu=qu, 
                t=t, f1=f, f2=f, d=d, 
                scheme=scheme, N=dict_dt['dt_all_orig'].shape[0], 
                verbose=True):    
    """Plots the ATT as a function of dose level for each coordinate 
    of a bivariate outcome. 

    Parameters
    ----------
    mean_eff_est : torch.Tensor
        Estimated ATT at all the observed treatment/dose levels. 
        This is extracted from the output dictionary of `estimate_mean_effects()`.
    mean_eff_true : torch.Tensor
        Ground truth ATT at all the observed treatment/dose levels.
    ql : torch.Tensor
        Lower quantiles of the estimated ATT at the observed treatment levels.
        This is extracted from the output dictionary of `estimate_mean_effects()`, 
        where the quantile is specified. 
    qu : torch.Tensor
        Upper quantiles of the estimated ATT at the observed treatment levels.
        This is extracted from the output dictionary of `estimate_mean_effects()`, 
        where the quantile is specified. 
    t : torch.Tensor
        Treatment/dose levels in all observations. 
    f1 : function
        The projection of the ground truth ATT curve to the first coordinate. 
    f2 : function
        The projection of the ground truth ATT curve to the second coordinate. 
    d : int
        The outcome dimension, by default d 
    scheme : str
        The design of the numerical experiment(s). This is just an identifier.  
    N : int
        The total number of observations aggregated over all observed treatment levels 
        *and* across time. 
        For example, if there are `n` observations at time = 0,
        then `N = 2 * n`.
    verbose : bool, optional
        Whether or not to display additional information on the plot. 
        If True, then the MSE, `scheme`, and `N` are printed on the plot. 
    
    Notes
    -----
    This function should only be used for bivariate outcomes (i.e., `d=2`).
    """
    x_grid = torch.linspace(0, 1, 500, device=t.device)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    # Seaborn default "deep" palette
    blue, orange = sns.color_palette("deep")[0], sns.color_palette("deep")[1]
    ql, qu = ql.cpu(), qu.cpu()
    ts = t.detach().cpu()
    alpha = 0.7
    q_width = 0.7

    for ax, col, f in zip(axes, range(d), (f1, f2)):
        y = mean_eff_est[:, col]        

        ax.plot(ts, y.detach().cpu(),
                color=orange, linestyle="--",
                label=fr'$\pi_{col + 1}\circ\hat{{f}}(\tau)$')
        ax.plot(ts, ql[:, col], color=orange, linestyle="--", 
                alpha=alpha, linewidth=q_width)
        ax.plot(ts, qu[:, col], color=orange, linestyle="--", 
                alpha=alpha, linewidth=q_width)
        ax.plot(x_grid.detach().cpu(), f(x_grid).detach().cpu(),
                color=blue,
                label=fr'$\pi_{col + 1}\circ f(\tau)$')

        ax.set_xlim(0, 1)
        ax.set_xlabel(fr'$\tau$')
        # ax.set_ylabel("Mean treatment effect on the treated")
        ax.legend()
    
    if verbose: 
        mse = ((mean_eff_est - mean_eff_true) ** 2).mean()
        fig.suptitle(
            rf"scheme = {scheme}; N = {N}; MSE = {mse:.04f}"
        )
    plt.tight_layout()


plot_effect(verbose=True)
plt.savefig(rf'{dir_plots}/{proj}-04curves-verbose.png',
            bbox_inches="tight")
plt.show()

plot_effect(verbose=False)
plt.savefig(rf'{dir_plots}/{proj}-03curves-concise.png', bbox_inches="tight")
plt.show()


# Compute MSE and put them in one place
dir_mses = "all-cont-" + scheme + "-mses" + rf'-v{version:02d}'
dir_mses
os.makedirs(dir_mses, exist_ok=True)


mse = ((mean_eff_est - mean_eff_true) ** 2).mean()
torch.save(mse, Path(dir_mses) / rf'{proj}-mse.pt')

