
# Cocycle applied to continuous treatments
import argparse
import torch
import numpy as np
import zuko
import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path
import random
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# Load source functions
from source.kernels_new import median_heuristic  # Written by Hugh Dance
from source.miscv00 import *
from source.cmmdulossv00 import *
from source.funsplotv00 import *

# ---- Housekeeping ----

parser = argparse.ArgumentParser()
parser.add_argument("--scheme", type=str, required=True)
parser.add_argument("--effect_type", type=str, required=True)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--script_ver", type=int, required=True)
args = parser.parse_args()


scheme = args.scheme
effect_type = args.effect_type
n_arm = args.n
version = args.script_ver

# n_arm = 1000
# effect_type = "id"
# version = 1
# scheme = "A"


loc = torch.tensor([0., 0])
cov = torch.tensor([1., -0.5, -0.5, 1]).reshape(-1, 2) / 100
noise = "indpt"

f = give_f(effect_type)

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

# True natural trend map represented as a matrix
p_matrix = torch.tensor([10, 0.,
                         -5,
                         10]).reshape(-1, 2)

def p(y, inv=False):
    return (p_matrix @ y.unsqueeze(-1))\
        .squeeze(-1) if not inv else (p_matrix.inverse() @ y.unsqueeze(-1))\
        .squeeze(-1)


d = 2 # Outcome dimension, i.e., R^d
d_cond_var = 2 # dimension of the conditioning variable 


def draw_samples(seed, 
                 receive_treatment, 
                 loc=loc, cov=cov, d=d, 
                 n_grp=n_arm, 
                 effect_type=effect_type,
                 p=p, # Natural trend, 
                 noise="indpt"): 
    # Defensive lines
    if noise not in ["indpt", "fixed"]:
        raise ValueError('noise must be either "indpt" or "fixed"')
    if effect_type not in ["id", "sqrt"]:
        raise ValueError('effect_type must be "id" or "sqrt".')
    # Forces the user to supply a seed
    torch.manual_seed(seed)  

    t = torch.rand(n_grp) # uniformly sample treatment from [0, 1]    
    loc_new = t.unsqueeze(-1) + loc # Create new mean vectors via broadcasting

    def get_treated_outcome(): 
        dt = torch.empty([n_grp, d])
        for i in range(loc_new.shape[0]):
            mvt_trt = torch.distributions.MultivariateNormal(loc=loc_new[i, :],
                                                            covariance_matrix=cov)
            dt[i, :] = mvt_trt.sample()
        return dt
    
    dt_outcome0 = get_treated_outcome() # Outcome at time = 0
    tmp = dt_outcome0 if noise == "fixed" else get_treated_outcome()    
    
    tau = t.unsqueeze(-1) # Treatment levels
    outcome_under_p = p(tmp) # True counterfactual

    # Outcome at time = 1; depends on treatment status
    effect = (t if effect_type == "id" else torch.sqrt(t))\
        .unsqueeze(-1) if receive_treatment else 0
    dt_outcome1 = outcome_under_p + effect       
    
    # Observtions at times = 0 and 1
    dt0 = torch.cat(
        [dt_outcome0, tau, torch.zeros([n_grp, 1])], dim=-1
    )
    dt1 = torch.cat(
        [dt_outcome1, tau, torch.ones([n_grp, 1])], dim=-1
    )
    return [dt.detach() for dt in [dt0, dt1]]


def prepare_train_test_data(lst_dt, prop_validate=0.2, 
                            d=d, d_cond_var=d_cond_var):    
    # All observations; a tensor; original scale
    dt_all_orig = torch.cat(lst_dt, dim=-2) 
    
    # Parameters of the "overall" affine transformation/standardization that 
    # standardize the *combined* data set
    param_std_overall = standardize_sample(dt_all_orig[:, :d], keep=True)
    # List of paramters of affine transformation that standardizes each data set
    lst_param_std_each = [standardize_sample(dt[:, :d], keep=True) 
                          for dt in lst_dt]
    
    # Reconstruct the affine transformation on the combined data set
    def std(dt, rst_std=param_std_overall):   
        outcome = (rst_std['scale'].inverse() @ (dt[:, :d] - rst_std['loc'])\
                   .unsqueeze(-1))\
                    .squeeze(-1)
        return torch.cat([outcome, dt[:, -d_cond_var:]], dim=-1)
    
    # torch.allclose(std(dt_all_orig), dt_all_std)

    # Observations in the training/validation set; a list; original scale
    lst_train_orig = [dt[:-int(dt.shape[0] * prop_validate), :] for dt in lst_dt]
    lst_test_orig = [dt[-int(dt.shape[0] * prop_validate):, :]
                     for dt in lst_dt]

    # Observations in the training/validation set; a list; 
    # the combined data set have zero mean and identity variance
    lst_train_std_overall = [std(dt) for dt in lst_train_orig]
    lst_test_std_overall = [std(dt) for dt in lst_test_orig]

    # Observations in the training/validation set; a list;
    # each individual data set has zero mean and identity variance
    lst_train_std_each = [
        torch.cat([standardize_sample(dt[:, :d]), dt[:, -d_cond_var:]], dim=-1)
        for dt in lst_train_orig
    ]
    lst_test_std_each = [
        torch.cat([standardize_sample(dt[:, :d]), dt[:, -d_cond_var:]], dim=-1) 
        for dt in lst_test_orig
    ]

    # Observations in the training/validation set; a tensor; original scale
    dt_train_orig = torch.cat(lst_train_orig, dim=-2)
    dt_test_orig = torch.cat(lst_test_orig, dim=-2)

    # Observations in the trainng/validation set; a tensor; 
    # with zero mean and identity variance
    dt_train_std_overall = torch.cat(lst_train_std_overall, dim=-2)
    dt_test_std_overall = torch.cat(lst_test_std_overall, dim=-2)

    # Observations in the training/validation set; a tensor'
    dt_train_std_each = torch.cat(lst_train_std_each, dim=-2)
    dt_test_std_each = torch.cat(lst_test_std_each, dim=-2)

    return {
        'dt_all_orig': dt_all_orig, 
        'param_std_overall': param_std_overall, # a dictionary 
        'lst_param_std_each': lst_param_std_each, # dictionary of dictionaries
        'lst_train_orig': lst_train_orig, 
        'lst_test_orig': lst_test_orig, 
        'lst_train_std_overall': lst_train_std_overall, 
        'lst_test_std_overall': lst_test_std_overall, 
        'lst_train_std_each': lst_train_std_each,
        'lst_test_std_each': lst_test_std_each,
        'dt_train_orig': dt_train_orig,
        'dt_test_orig': dt_test_orig, 
        'dt_train_std_overall': dt_train_std_overall,
        'dt_test_std_overall': dt_test_std_overall, 
        'dt_train_std_each': dt_train_std_each, 
        'dt_test_std_each': dt_test_std_each
    }



# # Draw samples for Group 0 and Group 1
# # We use Group 0 to estimate the map p
# # and build a cocycle with Group 1
# n_group0 = n_arm if n_arm < 1000 else 1000
n_group0 = n_arm
lst_dt_group0 = draw_samples(seed=0, n_grp=n_group0, receive_treatment=False)
lst_dt_group1 = draw_samples(seed=1, n_grp=n_arm, receive_treatment=True)


# Create a dictionary of original and standardized samples for each Group
dict_dt_group0 = prepare_train_test_data(lst_dt_group0)
dict_dt_group1 = prepare_train_test_data(lst_dt_group1)

# Extract the location-scale parameters; save them later for standardization
loc_scale_group1 = dict_dt_group1['param_std_overall']
loc_scale_group1.keys()

loc_scale_group0 = dict_dt_group0['lst_param_std_each']

for x in loc_scale_group0: 
    print(x.keys())



## [Section]: Plot standardized training data for groups 0 and 1

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
contour_xlim, contour_ylim = (-5, 5), (-5, 5)
plt.setp(axes, xlim=contour_xlim, ylim=contour_ylim)

for ax, dt, title in zip(axes.flat, dict_dt_group0['lst_train_std_each'],
                         [r"$s^{(0)}_{(\tau, 0)}$", r"$s^{(0)}_{(\tau, 1)}$"]): 
    plot_contour_kde(dt=dt[:, :d], col="Blues", ax=ax, scatter=False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(title)

fig.set_size_inches((15, 10))
plt.savefig(rf'{dir_plots}/{proj}-00-group0-samples-g0.png', bbox_inches="tight")
plt.show()


sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
contour_xlim, contour_ylim = (-5, 5), (-5, 5)
plt.setp(axes, xlim=contour_xlim, ylim=contour_ylim)

for ax, dt, title in zip(axes.flat, dict_dt_group1['lst_train_std_overall'],
                         [r"$s^{(1)}_{(\tau, 0)}$", r"$s^{(1)}_{(\tau, 1)}$"]):
    plot_contour_kde(dt=dt[:, :d], col="Blues", ax=ax, scatter=False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(title)

fig.set_size_inches((15, 10))
plt.savefig(rf'{dir_plots}/{proj}-00-group1-samples-g1.png', bbox_inches="tight")
plt.show()


## [Section]: Create bundles

# Create training/validation data sets and put them in the correct device
def create_tensordata(lst_outcome, lst_cond_var, device=device):
    return TensorDataset(
        torch.cat(lst_outcome, dim=-2).to(device).float(),
        torch.cat(lst_cond_var, dim=-2).to(device).float()
    )


# We must use the standardized data here
def get_dt_train_test(dict_dt, 
                      key_train_std, key_test_std, 
                      d=d, d_cond_var=d_cond_var): 
    lst_train_std = dict_dt[key_train_std]
    lst_test_std = dict_dt[key_test_std]    

    lst_outcome_train = [dt[:, :d] for dt in lst_train_std]
    lst_cond_var_train = [dt[:, -d_cond_var:] for dt in lst_train_std]
    lst_outcome_test = [dt[:, :d] for dt in lst_test_std]
    lst_cond_var_test = [dt[:, -d_cond_var:] for dt in lst_test_std]

    dt_train = create_tensordata(lst_outcome_train, lst_cond_var_train)
    dt_test = create_tensordata(lst_outcome_test, lst_cond_var_test)

    return dt_train, dt_test

# Create training and test data for groups 0 and 1
dt_train0, dt_test0 = get_dt_train_test(dict_dt=dict_dt_group0, 
                                        key_test_std='lst_test_std_each', 
                                        key_train_std='lst_train_std_each', 
                                        d_cond_var=1)

dt_train1, dt_test1 = get_dt_train_test(dict_dt=dict_dt_group1, 
                                        key_test_std='lst_test_std_overall', 
                                        key_train_std='lst_train_std_overall')


# ## [Section]: Write training loop


# Define two conditional normalizing flows
torch.manual_seed(0)
flows0 = zuko.flows.NSF(features=2, context=1,
                             transforms=1, bins=2, 
                             hidden_features=(64,))
flows0 = flows0.to(device)

flows1 = zuko.flows.NSF(features=2, context=d_cond_var,
                               transforms=3, bins=8,
                               hidden_features=(128, 128,))
flows1 = flows1.to(device)

# Define optimizers
opt0 = torch.optim.Adam(flows0.parameters(), lr=0.001)
opt1 = torch.optim.Adam(flows1.parameters(), lr=0.001)

# Define schedulers; not really used in this script
# ---- Training loop
torch.manual_seed(1)  
scheduler0 = torch.optim.lr_scheduler.StepLR(
    opt0, step_size=10 ** 9, gamma=1.0)
scheduler1 = torch.optim.lr_scheduler.StepLR(
    opt1, step_size=10 ** 9, gamma=1.0)


# ---- Median heuristic
print(inspect.getsource(median_heuristic))  # Hugh's implementation

# Set bandwidth for Gaussian kernerl; subsample to avoid memory overload

def get_bandwidth(dt, d=d):
    torch.manual_seed(0)  # Reproducible subsampling
    bandwidth = median_heuristic(X=dt[:, :d]) \
        if dt.shape[0] <= 6000 \
        else median_heuristic(X=subsample(dt)[:, :d])
    bandwidth = bandwidth.to(device)
    return bandwidth

bandwidth0 = get_bandwidth(dict_dt_group0['dt_train_std_each'])
bandwidth1 = get_bandwidth(dict_dt_group1['dt_train_std_overall'])
print(rf"bandwidth0: {bandwidth0}")
print(rf"bandwidth1: {bandwidth1}")


@dataclass
class DataBundle:
    dict_dt: dict
    loc_scale: dict


bundle_dat0 = DataBundle(dict_dt=dict_dt_group0,
                         loc_scale=loc_scale_group0)

bundle_dat1 = DataBundle(dict_dt=dict_dt_group1,
                         loc_scale=loc_scale_group1)

@dataclass
class TrainBundle:
    data: object
    flows: zuko.flows
    opt: torch.optim.Optimizer
    scheduler: object
    bandwidth: torch.tensor
    ckpt_dir: str
    saved_vars_dir: str
    dt_train: torch.tensor
    dt_test: torch.tensor

# Create directories for checkpoints
ckpt_dir0 = "ckpts" + proj + "-g0"
ckpt_dir0
ckpt_dir1 = "ckpts" + proj + '_g1'
ckpt_dir1

os.makedirs(ckpt_dir0, exist_ok=True)
os.makedirs(ckpt_dir1, exist_ok=True)

# Create directories for saved variables
dir_saved_vars0 = "proj-vars" + proj + '-g0' 
dir_saved_vars0

dir_saved_vars1 = "proj-vars" + proj + '-g1'
dir_saved_vars1

os.makedirs(dir_saved_vars0, exist_ok=True)
os.makedirs(dir_saved_vars1, exist_ok=True)

bundle_train0 = TrainBundle(
    data=bundle_dat0, 
    flows=flows0, 
    opt=opt0, 
    scheduler=scheduler0, 
    bandwidth=bandwidth0, 
    ckpt_dir=ckpt_dir0, 
    saved_vars_dir=dir_saved_vars0, 
    dt_train=dt_train0, 
    dt_test=dt_test0
)

bundle_train1 = TrainBundle(
    data=bundle_dat1, 
    flows=flows1, 
    opt=opt1, 
    scheduler=scheduler1, 
    bandwidth=bandwidth1, 
    ckpt_dir=ckpt_dir1, 
    saved_vars_dir=dir_saved_vars1, 
    dt_train=dt_train1, 
    dt_test=dt_test1
)


# ## [Section]: Set up checkpoints


# -------------------- config --------------------
EPOCHS = 1000
CKPT_STRIDE = 10  # save a full checkpoint every CKPT_STRIDE epochs
BASE_SEED = 123
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 800



# -------------------- seeding --------------------


def seed_all(s: int):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def begin_epoch_seed(epoch: int):
    # keep per-epoch randomness deterministic (dropout, data aug, etc.)
    s = BASE_SEED + epoch
    seed_all(s)


seed_all(BASE_SEED)  # initial/global seed

# -------------------- data loaders --------------------


def make_train_loader(dataset, epoch):
    # Deterministic shuffling via per-epoch generator; num_workers=0 keeps it simple.
    g = torch.Generator()
    g.manual_seed(BASE_SEED + epoch)
    return DataLoader(dataset, batch_size=BATCH_SIZE_TRAIN, 
                      shuffle=True, generator=g)


def make_test_loader(dataset, epoch):
    # Deterministic shuffling via per-epoch generator; num_workers=0 keeps it simple.
    g = torch.Generator()
    g.manual_seed(BASE_SEED + epoch)
    return DataLoader(dataset, batch_size=BATCH_SIZE_TEST, 
                      shuffle=False)


# # ---- Training loop
# torch.manual_seed(1)  # To ensure reproducibility
# scheduler = torch.optim.lr_scheduler.StepLR(
#     opt, step_size=10 ** 9, gamma=1.0)


# -------------------- train / eval --------------------


def train_one_epoch(epoch, epochs, flows, opt, loader, bandwidth):
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
    cmmd_u_loss = 0.
    for y_batch, cond_var_batch in loader:
        loss = compute_cmmdu(y_batch, cond_var_batch, bandwidth, flows)        
        cmmd_u_loss += loss.item()

    if epoch in (0, epochs - 1) or (epoch % 100 == 0):
        print(
            f"Epoch {epoch} validation_loss: {cmmd_u_loss / len(loader):.4f}"
        )
    return cmmd_u_loss / len(loader)



# -------------------- checkpoint I/O --------------------

bundle = bundle_train0
bundle.ckpt_dir

def ckpt_path(bundle, epoch):
    return f"{bundle.ckpt_dir}/full_{epoch:04d}.pt"


def save_full_ckpt(bundle, epoch, loss_training=None, loss_validation=None):
    ckpt = {
        "epoch": epoch,
        "flows": bundle.flows.state_dict(),
        "optimizer": bundle.opt.state_dict(),
        "scheduler": bundle.scheduler.state_dict(),
    }

    if loss_training is not None:
        ckpt["loss_training"] = loss_training
    if loss_validation is not None:
        ckpt["loss_validation"] = loss_validation

    torch.save(ckpt, ckpt_path(bundle, epoch))


def load_full_ckpt(bundle, epoch, device=device):
    ckpt = torch.load(ckpt_path(bundle, epoch), map_location=device)
    bundle.flows.load_state_dict(ckpt["flows"])
    bundle.opt.load_state_dict(ckpt["optimizer"])
    bundle.scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


# -------------------- workflows --------------------


def train_all(bundle,
              epochs=EPOCHS,
              ckpt_stride=CKPT_STRIDE,
              start_epoch=0,
              loss_training=None,
              loss_validation=None):
    if loss_training is None:
        loss_training = torch.empty(epochs)
    if loss_validation is None:
        loss_validation = torch.empty(epochs)    

    for epoch in range(start_epoch, epochs):
        begin_epoch_seed(epoch)

        # Training
        train_loader = make_train_loader(dataset=bundle.dt_train, epoch=epoch)
        loss_training[epoch] = train_one_epoch(
            epoch=epoch,
            epochs=epochs,
            flows=bundle.flows,
            opt=bundle.opt,
            loader=train_loader,
            bandwidth=bundle.bandwidth
        )
        bundle.scheduler.step()

        # Validation
        test_loader = make_test_loader(dataset=bundle.dt_test, epoch=epoch)
        loss_validation[epoch] = test_one_epoch(
            epoch=epoch,
            epochs=epochs,
            flows=bundle.flows,
            loader=test_loader,
            bandwidth=bundle.bandwidth
        )

        if (epoch % ckpt_stride == 0) or epoch == (epochs - 1):
            save_full_ckpt(
                bundle=bundle,
                epoch=epoch,
                loss_training=loss_training,
                loss_validation=loss_validation
            )

    return loss_training, loss_validation


print(EPOCHS)


## [Section]: Execute training loop

# Training is time-comsuming so we save training and validation losses
# inside every checkpoint
# Next time the script is run it will automatically load the
# saved full losses if they are available or resume from the lastest checkpoint 
# If no checkpoint exists it starts the training loop from scratch

def run_training_loop(bundle, epochs=EPOCHS):
    final_losses_path = Path(bundle.saved_vars_dir) / "losses.pt"

    # If final losses exist, training already finished
    if final_losses_path.exists():
        losses = torch.load(final_losses_path, map_location='cpu')
        # Indicate the training process is complete
        (Path(bundle.saved_vars_dir) / "training_complete.flag").touch()
        return losses["loss_training"].cpu(), losses["loss_validation"].cpu()

    # Find latest checkpoint
    pattern = re.compile(r"full_(\d{4})\.pt$")
    saved_epochs = []
    for name in os.listdir(bundle.ckpt_dir):
        m = pattern.match(name)
        if m:
            saved_epochs.append(int(m.group(1)))

    # No checkpoint found -> start from scratch
    if not saved_epochs:
        loss_training, loss_validation = train_all(bundle, epochs=epochs)

    else:
        last_epoch = max(saved_epochs)
        ckpt = load_full_ckpt(bundle, last_epoch)
        loss_training = ckpt["loss_training"].cpu()
        loss_validation = ckpt["loss_validation"].cpu()

        loss_training, loss_validation = train_all(
            bundle=bundle,
            epochs=epochs,
            start_epoch=last_epoch + 1,
            loss_training=loss_training,
            loss_validation=loss_validation
        )

    # Save final losses once training is complete
    torch.save(
        {
            "loss_training": loss_training.cpu(),
            "loss_validation": loss_validation.cpu(),
        },
        final_losses_path
    )

    # Indicate the training process is complete
    (Path(bundle.saved_vars_dir) / "training_complete.flag").touch()

    return loss_training.cpu(), loss_validation.cpu()


# Compute losses if not already 
loss_training_g0, loss_validation_g0 = run_training_loop(bundle_train0, epochs=500)
loss_training_g1, loss_validation_g1 = run_training_loop(bundle_train1)

# Note that loss_training and loss_validation are on the CPU, NOT EVER on the GPU
[x.device for x in [loss_training_g0, loss_validation_g1]]


# ## [Section]: Plot losses

plot_loss(loss_training_g0, "Training loss (Trend map)")
plt.savefig(rf'{dir_plots}/{proj}-01train-g0.png', bbox_inches="tight")

plot_loss(loss_validation_g0, "Validation loss (Trend map)")
plt.savefig(rf'{dir_plots}/{proj}-02validation-g0.png', bbox_inches="tight")

plot_loss(loss_training_g1, "Training loss (cocycle)")
plt.savefig(rf'{dir_plots}/{proj}-01train-g1.png', bbox_inches="tight")

plot_loss(loss_validation_g1, "Validation loss (cocycle)")
plt.savefig(rf'{dir_plots}/{proj}-02validation-g1.png', bbox_inches="tight")


# ## [Section]: Recover the flows based on checkpoints


def recover_flows(bundle,
                  n,
                  epochs=EPOCHS,
                  ckpt_stride=CKPT_STRIDE):
    """
    Reconstruct and save weights at epoch n.
    This version does NOT rely on a global EPOCHS constant.
    It uses the actually saved full checkpoints to determine feasibility.
    """
    # Do not recover flows unless the training is complete
    complete_flag = Path(bundle.saved_vars_dir) / "training_complete.flag"
    if not complete_flag.exists():
        raise RuntimeError(
            rf"Training incomplete for {bundle.saved_vars_dir}; cannot recover flows.")
    
    if torch.is_tensor(n):
        n = int(n.item())
    else:
        n = int(n)

    # Fast path: exact checkpoint exists
    s = int(np.floor(n / ckpt_stride) * ckpt_stride)
    pattern = re.compile(rf"full_{n:04d}\.pt")
    checkpoints = os.listdir(bundle.ckpt_dir)
    if n == s or n == (epochs - 1) or any(pattern.match(x) for x in checkpoints):
        load_full_ckpt(bundle, n)
        print(
            f"Full checkpoint already saved for epoch {n} (no replay) -> {ckpt_path(bundle, n)}")
        return

    # Load s and replay to n
    load_full_ckpt(bundle, s)
    for epoch in range(s + 1, n + 1):
        begin_epoch_seed(epoch)
        train_loader = make_train_loader(dataset=bundle.dt_train, epoch=epoch)
        train_one_epoch(
            epoch=epoch, epochs=epochs,
            flows=bundle.flows, opt=bundle.opt,
            loader=train_loader, bandwidth=bundle.bandwidth)
        bundle.scheduler.step()

    save_full_ckpt(bundle, n)
    print(
        f"Saved checkpoint for epoch {n} (replayed {n - s} epoch(s))")


recover_flows(bundle=bundle_train0, n=loss_validation_g0.argmin())
recover_flows(bundle=bundle_train1, n=loss_validation_g1.argmin())

# Reconstruct the trend map
# Estimated natural trend function on the *original* scale

def trend(y, 
          bundle_dat=bundle_dat0,
          bundle_train=bundle_train0, 
          device=device):
    y = y.to(device)
    c_from = torch.tensor([0.], device=device)
    c_to = torch.tensor([1.], device=device)
    # Affine transformation; standardization and its inverse
    loc_scale = bundle_dat.loc_scale # Recall this is a dict of two dicts
    def affine(y, l_s, inv=False, device=device):
        l_s = {k: v.to(device) for k, v in l_s.items()}
        if inv:
            return (l_s['scale'] @ y.unsqueeze(-1)).squeeze(-1) + l_s['loc']
        else:
            return (l_s['scale'].inverse() @ (y - l_s['loc']).unsqueeze(-1)).squeeze(-1)
    
    # Trend map on *standardized samples* 

    def tmi(c_to, c_from, y_from, flows=bundle_train.flows):
       with torch.no_grad():
           return flows(c=c_to).transform.inv(
               flows(c=c_from).transform(y_from)
           )
    return affine(
        y=tmi(c_to, c_from,
              affine(y, l_s=loc_scale[0]), flows=bundle_train.flows),
        l_s=loc_scale[1],
        inv=True
    )
    

# Check if the trend map is corrected estimated

lst_tmp0 = [
    lst_dt_group0[1],
    trend(lst_dt_group0[0][:, :d])
]

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
contour_xlim, contour_ylim = (-15, 15), (-15, 15)
plt.setp(axes, xlim=contour_xlim, ylim=contour_ylim)

for ax, dt, title in zip(axes.flat, lst_tmp0,
                         [r"$s_{(0, 1)}$", r"$\hat{s}_{(0, 1)}$"]):
    plot_contour_kde(dt=dt[:, :d], col="Blues", ax=ax, scatter=False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(title)
fig.set_size_inches((15, 10))
plt.savefig(rf'{dir_plots}/{proj}-double-check.png', bbox_inches="tight")
plt.show()


# ## [Section]: Plot cocycles

# Sort data by increasing treatment value
def sort_by_treatment(dt): 
    indeces = dt[:, -d_cond_var:][:, 0].argsort()
    return dt[indeces, :]


# Cocycle: TMI transports between observations on the *original* scale

def cocycle(y, 
            bundle_dat=bundle_dat1,
            bundle_train=bundle_train1, 
            c_to=torch.tensor([0., 1]), 
            c_from=torch.tensor([0., 0]),          
            device=device):
    y = y.to(device)
    c_to = c_to.to(device)
    c_from = c_from.to(device)
    loc_scale = bundle_dat.loc_scale # Recall that this is a dict of tensors

    # Affine transformation; standardization and its inverse
    def affine(y, l_s, inv=False, device=device):
        l_s = {k: v.to(device) for k, v in l_s.items()}
        if inv:
            return (l_s['scale'] @ y.unsqueeze(-1)).squeeze(-1) + l_s['loc']
        else:
            return (l_s['scale'].inverse() @ (y - l_s['loc']).unsqueeze(-1)).squeeze(-1)
    # TMI transport maps on *standardized samples*
    def tmi(c_to, c_from, y_from, flows=bundle_train.flows):              
       with torch.no_grad():
            return flows(c=c_to).transform.inv(
                flows(c=c_from).transform(y_from)
            )
    return affine(
        y=tmi(c_to, c_from,
                  affine(y, l_s=loc_scale), flows=bundle_train.flows),
        l_s=loc_scale,
        inv=True
    )



lst_dt = lst_dt_group1

def estimate_mean_effects(lst_dt=lst_dt, d=d, device=device, 
                         trend=trend,
                         n_trted=n_arm,
                         cocycle=cocycle,
                         q_lower = 0.25, q_upper = 0.75, 
                         filename_mean_effects="mean_effects", 
                         filename_q_lower="quantile_lower", 
                         filename_q_upper="quantile_upper"):
    filenames = [filename_mean_effects, filename_q_lower, filename_q_upper]
     
    rst_dict = {}      
     
    mean_effects = torch.empty([n_trted, d], device=device)
    quantile_lower = torch.empty([n_trted, d], device=device)
    quantile_upper = torch.empty([n_trted, d], device=device)
    # Sorted data
    [sorted0, sorted1] = [sort_by_treatment(dt).to(device) for dt in lst_dt]   
    

    def move_data(dt_to_be_sent, df_trted, i):             
        tmp = cocycle(c_to=df_trted[i, -d_cond_var:],                         
                      c_from=dt_to_be_sent[:, -d_cond_var:],
                      y=dt_to_be_sent[:, :d])
        return tmp        

    for i in range(n_trted):
        with torch.no_grad():
            tmp0 = move_data(dt_to_be_sent=sorted0, 
                             df_trted=sorted0,
                             i=i)
            # Esimated counterfactual distribution
            est_counterfactual = trend(tmp0)
            # Estimated observed distribution 
            est_observed = move_data(
                dt_to_be_sent=sorted1,                 
                df_trted=sorted1, 
                i=i
            )
            diff = est_observed - est_counterfactual

        quantile_lower[i, :] = torch.quantile(diff, q_lower, dim=0)
        quantile_upper[i, :] = torch.quantile(diff, q_upper, dim=0)                
        mean_effects[i, :] = diff.mean(dim=-2)

        to_be_saved = [mean_effects, quantile_lower, quantile_upper]
    for i in range(len(filenames)):
        rst_dict[filenames[i]] = to_be_saved[i]

    return rst_dict


dict_mean_effects = estimate_mean_effects()
mean_eff_est = dict_mean_effects['mean_effects']
ql = dict_mean_effects['quantile_lower']
qu = dict_mean_effects['quantile_upper']



# Sorted data
[sorted0, sorted1] = [sort_by_treatment(dt).to(device) for dt in lst_dt]

# Sniff check; should be true
torch.allclose(sorted0[:, -d_cond_var:][:, 0], 
               sorted1[:, -d_cond_var:][:, 0])

t = sorted1[:, -d_cond_var:][:, 0]

mean_eff_true = f(t).unsqueeze(-1).expand(len(t), 2)

def plot_effect(mean_eff_est=mean_eff_est, 
                mean_eff_true=mean_eff_true,
                ql=ql, qu=qu, 
                t=t, f1=f, f2=f, d=d, 
                scheme=scheme, 
                N=dict_dt_group1['dt_all_orig'].shape[0], 
                verbose=True):    
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

