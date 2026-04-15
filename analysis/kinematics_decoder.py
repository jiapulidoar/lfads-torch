import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def build_lagged_features(neural_features, lag_ms, bin_size_ms):
    """
    Lag neural features by a fixed offset to account for
    neural-to-kinematic delay.

    Args:
        neural_features : (n_trials, n_time, n_features)
        lag_ms          : lag in ms (90 ms from paper)
        bin_size_ms     : bin size in ms (20 ms from paper)
    Returns:
        lagged           : (n_trials, n_time, n_features)
    """
    lag_bins = lag_ms // bin_size_ms       # 90ms / 20ms = 4 bins
    if lag_bins == 0:
        return neural_features
    # Shift features forward in time (neural leads kinematics)
    # Pad the beginning with the first frame to avoid edge artifacts
    pad    = np.repeat(neural_features[:, :1, :], lag_bins, axis=1)
    lagged = np.concatenate([pad, neural_features[:, :-lag_bins, :]], axis=1)
    return lagged


def bin_features(neural_features, original_bin_ms, target_bin_ms):
    """
    Rebin neural features from original_bin_ms to target_bin_ms.

    Args:
        neural_features : (n_trials, n_time, n_features)
        original_bin_ms : current bin size in ms
        target_bin_ms   : desired bin size in ms (20 ms from paper)
    Returns:
        binned          : (n_trials, n_time_rebinned, n_features)
    """
    factor = target_bin_ms // original_bin_ms
    if factor <= 1:
        return neural_features
    n_trials, n_time, n_feat = neural_features.shape
    # Trim to multiple of factor
    n_trim  = (n_time // factor) * factor
    trimmed = neural_features[:, :n_trim, :]
    # Reshape and average within each bin
    binned  = trimmed.reshape(n_trials, n_trim // factor, factor, n_feat).mean(axis=2)
    return binned


def extract_decoding_window(neural_features, kinematics,
                             movement_onset_idx,
                             pre_onset_ms, post_onset_ms,
                             bin_size_ms):
    """
    Extract the decoding window around movement onset:
    -250 ms to +450 ms as described in the paper.

    Args:
        neural_features   : (n_trials, n_time, n_features)
        kinematics        : (n_trials, n_time, 2)  — vx, vy
        movement_onset_idx: (n_trials,) index of movement onset per trial
        pre_onset_ms      : ms before onset to include (250)
        post_onset_ms     : ms after onset to include  (450)
        bin_size_ms       : bin size in ms
    Returns:
        X : (n_trials, window_bins, n_features)
        Y : (n_trials, window_bins, 2)
    """
    pre_bins  = pre_onset_ms  // bin_size_ms    # 250/20 = 12 bins
    post_bins = post_onset_ms // bin_size_ms    # 450/20 = 22 bins
    win_len   = pre_bins + post_bins            # 34 bins total

    X_list, Y_list = [], []
    for i, onset in enumerate(movement_onset_idx):
        start = onset - pre_bins
        end   = onset + post_bins
        if start < 0 or end > neural_features.shape[1]:
            continue
        X_list.append(neural_features[i, start:end, :])
        Y_list.append(kinematics[i,      start:end, :])

    return np.array(X_list), np.array(Y_list)    # each (n_valid, win_len, *)


class OLEDecoder:
    """
    OLE decoder that predicts vx, vy at every timestep.
    Input  : (n_trials, n_time, n_features)
    Output : (n_trials, n_time, 2)
    """
    def __init__(self, alpha=1.0):
        self.alpha  = alpha
        self.models = {}
        self.is_fit = False

    def fit(self, X, Y):
        """
        Args:
            X : (n_trials, n_time, n_features)
            Y : (n_trials, n_time, 2)  — vx, vy at every timestep
        """
        # Flatten trials and time together → one sample per (trial, timestep)
        n_trials, n_time, n_feat = X.shape
        X_flat = X.reshape(n_trials * n_time, n_feat)       # (n_trials*n_time, n_feat)
        Y_flat = Y.reshape(n_trials * n_time, 2)            # (n_trials*n_time, 2)

        for dim, name in enumerate(["vx", "vy"]):
            model = Ridge(alpha=self.alpha)
            model.fit(X_flat, Y_flat[:, dim])
            self.models[name] = model
        self.is_fit = True

    def predict(self, X):
        """
        Args:
            X : (n_trials, n_time, n_features)
        Returns:
            Y_hat : (n_trials, n_time, 2)
        """
        assert self.is_fit
        n_trials, n_time, n_feat = X.shape
        X_flat = X.reshape(n_trials * n_time, n_feat)

        vx = self.models["vx"].predict(X_flat).reshape(n_trials, n_time)
        vy = self.models["vy"].predict(X_flat).reshape(n_trials, n_time)
        return np.stack([vx, vy], axis=-1)                  # (n_trials, n_time, 2)


def cross_validated_decoding(X, Y, n_splits=5, alpha=1.0):
    """
    5-fold CV decoding — now predicts full velocity time series.

    Args:
        X : (n_trials, n_time, n_features)
        Y : (n_trials, n_time, 2)
    Returns:
        r2_x, r2_y  : R² computed across all timepoints and trials
        y_true      : (n_trials, n_time, 2)
        y_pred      : (n_trials, n_time, 2)
    """
    kf         = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        decoder = OLEDecoder(alpha=alpha)
        decoder.fit(X_train, Y_train)

        y_hat = decoder.predict(X_val)              # (n_val, n_time, 2)
        y_true_all.append(Y_val)
        y_pred_all.append(y_hat)

    y_true = np.concatenate(y_true_all, axis=0)    # (n_trials, n_time, 2)
    y_pred = np.concatenate(y_pred_all, axis=0)

    # R² across all trials and timepoints flattened
    r2_x = r2_score(y_true[:, :, 0].ravel(), y_pred[:, :, 0].ravel())
    r2_y = r2_score(y_true[:, :, 1].ravel(), y_pred[:, :, 1].ravel())

    return r2_x, r2_y, y_true, y_pred


def integrate_velocity_to_trajectory(decoded_velocity, initial_position, bin_size_ms):
    """
    Reconstruct reach trajectory by integrating decoded velocity,
    seeded with the true initial position (as in paper Fig. 2d).

    Args:
        decoded_velocity : (n_time, 2)  — vx, vy in units/ms
        initial_position : (2,)         — true x0, y0
        bin_size_ms      : bin size in ms (for correct scaling)
    Returns:
        trajectory       : (n_time, 2)  — x, y positions
    """
    dt         = bin_size_ms / 1000.0          # convert ms → seconds
    n_time     = decoded_velocity.shape[0]
    trajectory = np.zeros((n_time, 2))
    trajectory[0] = initial_position

    for t in range(1, n_time):
        trajectory[t] = trajectory[t-1] + decoded_velocity[t-1] * dt

    return trajectory


def run_decoding_pipeline_prealigned(
    neural_features,
    kinematics,
    initial_positions,
    bin_size_original_ms = 5,
    bin_size_target_ms   = 20,
    lag_ms               = 90,
    n_splits             = 5,
    alpha                = 1.0
):
    # 1. Rebin to 20 ms
    #features = bin_features(neural_features,
    #                        bin_size_original_ms, bin_size_target_ms)
    kin      = bin_features(kinematics,
                            bin_size_original_ms, bin_size_target_ms)

    # 2. Apply 90 ms lag
    features = build_lagged_features(features, lag_ms, bin_size_target_ms)

    # 3. Trim to [-250, +450] ms
    pre_excess_bins = (450 - 250) // bin_size_target_ms
    features = features[:, pre_excess_bins:, :]
    kin      = kin[:,      pre_excess_bins:, :]

    # 4. 5-fold CV decoding — now full time series
    r2_x, r2_y, y_true, y_pred = cross_validated_decoding(
        features, kin, n_splits=n_splits, alpha=alpha
    )
    # y_true, y_pred are now (n_trials, n_time, 2)

    # 5. Integrate full velocity time series → curved trajectories
    dt           = bin_size_target_ms / 1000.0
    trajectories_pred = []
    trajectories_true = []

    for i in range(len(y_pred)):
        # Predicted
        pred_pos      = np.zeros((y_pred.shape[1], 2))
        pred_pos[0]   = initial_positions[i]
        for t in range(1, y_pred.shape[1]):
            pred_pos[t] = pred_pos[t-1] + y_pred[i, t-1] * dt
        trajectories_pred.append(pred_pos)

        # True
        true_pos      = np.zeros((y_true.shape[1], 2))
        true_pos[0]   = initial_positions[i]
        for t in range(1, y_true.shape[1]):
            true_pos[t] = true_pos[t-1] + y_true[i, t-1] * dt
        trajectories_true.append(true_pos)

    print(f"R² x-velocity : {r2_x:.3f}")
    print(f"R² y-velocity : {r2_y:.3f}")
    print(f"R² mean       : {(r2_x + r2_y) / 2:.3f}")

    return {
        "r2_x"              : r2_x,
        "r2_y"              : r2_y,
        "y_true"            : y_true,               # (n_trials, n_time, 2)
        "y_pred"            : y_pred,               # (n_trials, n_time, 2)
        "trajectories_pred" : trajectories_pred,    # list of (n_time, 2)
        "trajectories_true" : trajectories_true,    # list of (n_time, 2)
    }


# Utils 

def prepare_kinematics(kinematics, bin_size_ms=5):
    """
    Convert (3, n_time, n_trials) kinematics into (n_trials, n_time, 2)
    vx, vy using recorded speed and position-derived direction.

    This is more accurate than pure differentiation because it uses
    the recorded speed magnitude and only derives direction from positions.

    Args:
        kinematics  : (3, n_time, n_trials)  — x, y, speed
        bin_size_ms : bin size in ms
    Returns:
        kin_vel     : (n_trials, n_time, 2)  — vx, vy
        positions   : (n_trials, n_time, 2)  — x, y
    """
    dt    = bin_size_ms / 1000.0

    x     = kinematics[0]       # (n_time, n_trials)
    y     = kinematics[1]
    speed = kinematics[2]       # (n_time, n_trials) — scalar magnitude

    # Derive movement direction from positions
    dx = np.diff(x, axis=0)     # (n_time-1, n_trials)
    dy = np.diff(y, axis=0)

    # Pad to restore shape
    dx = np.concatenate([dx[:1, :], dx], axis=0)    # (n_time, n_trials)
    dy = np.concatenate([dy[:1, :], dy], axis=0)

    # Unit direction vector from positional differences
    norm    = np.sqrt(dx**2 + dy**2) + 1e-8         # avoid division by zero
    dir_x   = dx / norm                              # (n_time, n_trials)
    dir_y   = dy / norm

    # Scale unit direction by recorded speed → signed vx, vy
    vx = dir_x * speed
    vy = dir_y * speed

    # Transpose to (n_trials, n_time, 2)
    kin_vel   = np.stack([vx, vy], axis=-1).transpose(2, 0, 1)
    positions = np.stack([x,  y],  axis=-1).transpose(2, 0, 1)

    return kin_vel, positions


def get_initial_positions(positions, movement_onset_idx):
    """
    Extract the true initial position at movement onset per trial,
    used to seed trajectory integration.

    Args:
        positions         : (n_trials, n_time, 2)
        movement_onset_idx: (n_trials,)
    Returns:
        initial_positions : (n_trials, 2)
    """
    return np.array([
        positions[i, onset, :]
        for i, onset in enumerate(movement_onset_idx)
    ])





def plot_kinematic_predictions(results, kin_vel, bin_size_ms=20,
                                n_example_trials=5, trial_indices=None,
                                condition_labels=None):
    """
    Plot decoded kinematic predictions vs ground truth.

    Panels:
        Row 1 — vx and vy time series for a few example trials
        Row 2 — scatter of true vs predicted velocity (all validation trials)
        Row 3 — reconstructed reach trajectories vs true trajectories

    Args:
        results          : output dict from run_decoding_pipeline_prealigned
        kin_vel          : (n_trials, n_time, 2) ground truth velocities
                           at original bin size — will be rebinned to match
        bin_size_ms      : bin size of the decoded output (20 ms)
        n_example_trials : how many individual trials to show in row 1
        trial_indices    : specific trial indices to plot (overrides n_example)
        condition_labels : (n_trials,) optional — color trials by condition
    """
    y_true  = results["y_true"]         # (n_trials, 2)
    y_pred  = results["y_pred"]         # (n_trials, 2)
    trajs   = results["trajectories"]   # list of (n_time, 2)
    n_trials = y_true.shape[0]

    if trial_indices is None:
        trial_indices = np.linspace(0, n_trials - 1,
                                    n_example_trials, dtype=int)

    time_axis = np.arange(len(trajs[0])) * bin_size_ms - 450  # ms from onset

    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: vx and vy time series ─────────────────────────────────────────
    ax_vx = fig.add_subplot(gs[0, :2])
    ax_vy = fig.add_subplot(gs[0, 2])

    colors = plt.get_cmap("tab10", n_trials)

    for i, tidx in enumerate(trial_indices):
        col = colors(i)
        # True velocity — tile the single mean value across time for display
        vx_true = np.full(len(time_axis), y_true[tidx, 0])
        vy_true = np.full(len(time_axis), y_true[tidx, 1])
        vx_pred = np.full(len(time_axis), y_pred[tidx, 0])
        vy_pred = np.full(len(time_axis), y_pred[tidx, 1])

        ax_vx.plot(time_axis, vx_true, color=col, lw=1.5,
                   label=f"Trial {tidx}" if i == 0 else "")
        ax_vx.plot(time_axis, vx_pred, color=col, lw=1.5,
                   linestyle="--", alpha=0.8)
        ax_vy.plot(time_axis, vy_true, color=col, lw=1.5)
        ax_vy.plot(time_axis, vy_pred, color=col, lw=1.5,
                   linestyle="--", alpha=0.8)

    ax_vx.axvline(0, color="gray", lw=0.8, linestyle=":")
    ax_vy.axvline(0, color="gray", lw=0.8, linestyle=":")
    ax_vx.set_xlabel("Time from onset (ms)")
    ax_vx.set_ylabel("x velocity")
    ax_vy.set_xlabel("Time from onset (ms)")
    ax_vy.set_ylabel("y velocity")
    ax_vx.set_title("vx: true (—) vs predicted (--)")
    ax_vy.set_title("vy")

    # ── Row 2: true vs predicted scatter ─────────────────────────────────────
    ax_sx = fig.add_subplot(gs[1, 0])
    ax_sy = fig.add_subplot(gs[1, 1])
    ax_r2 = fig.add_subplot(gs[1, 2])

    # Color by condition if provided, otherwise single color
    if condition_labels is not None:
        conds      = np.unique(condition_labels)
        cmap_cond  = plt.get_cmap("tab10", len(conds))
        pt_colors  = [cmap_cond(np.where(conds == c)[0][0])
                      for c in condition_labels]
    else:
        pt_colors = ["steelblue"] * n_trials

    ax_sx.scatter(y_true[:, 0], y_pred[:, 0],
                  c=pt_colors, alpha=0.4, s=15, linewidths=0)
    ax_sy.scatter(y_true[:, 1], y_pred[:, 1],
                  c=pt_colors, alpha=0.4, s=15, linewidths=0)

    for ax, dim, label in zip([ax_sx, ax_sy],
                               [0, 1], ["vx", "vy"]):
        lims = [min(y_true[:, dim].min(), y_pred[:, dim].min()),
                max(y_true[:, dim].max(), y_pred[:, dim].max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)    # identity line
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"predicted {label}")
        ax.set_title(f"{label}  R²={results[f'r2_{label[-1]}']:.3f}")
        ax.set_aspect("equal")

    # R² bar summary
    r2_vals = [results["r2_x"], results["r2_y"],
               (results["r2_x"] + results["r2_y"]) / 2]
    bars    = ax_r2.bar(["R²x", "R²y", "R²mean"], r2_vals,
                        color=["steelblue", "coral", "gray"],
                        width=0.5, edgecolor="white")
    ax_r2.set_ylim(0, 1)
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("Decoding performance")
    for bar, val in zip(bars, r2_vals):
        ax_r2.text(bar.get_x() + bar.get_width() / 2,
                   val + 0.02, f"{val:.3f}",
                   ha="center", va="bottom", fontsize=9)

    # ── Row 3: reconstructed trajectories ────────────────────────────────────
    ax_traj_pred = fig.add_subplot(gs[2, 0])
    ax_traj_true = fig.add_subplot(gs[2, 1])
    ax_traj_both = fig.add_subplot(gs[2, 2])

    # Reconstruct true trajectories by integrating ground truth velocity
    dt = bin_size_ms / 1000.0
    true_trajs = []
    for tidx in range(n_trials):
        traj = np.zeros((len(trajs[tidx]), 2))
        traj[0] = trajs[tidx][0]                # same seed as decoded
        for t in range(1, len(traj)):
            traj[t] = traj[t-1] + np.array([
                y_true[tidx, 0], y_true[tidx, 1]
            ]) * dt
        true_trajs.append(traj)

    for i, tidx in enumerate(trial_indices):
        col        = colors(i)
        pred_traj  = np.array(trajs[tidx])
        true_traj  = np.array(true_trajs[tidx])

        ax_traj_pred.plot(pred_traj[:, 0], pred_traj[:, 1],
                          color=col, lw=1.2, alpha=0.9)
        ax_traj_pred.scatter(*pred_traj[0],  color=col, s=30, zorder=5)
        ax_traj_pred.scatter(*pred_traj[-1], color=col, s=50,
                             marker="*", zorder=5)

        ax_traj_true.plot(true_traj[:, 0], true_traj[:, 1],
                          color=col, lw=1.2, alpha=0.9)
        ax_traj_true.scatter(*true_traj[0],  color=col, s=30, zorder=5)
        ax_traj_true.scatter(*true_traj[-1], color=col, s=50,
                             marker="*", zorder=5)

        ax_traj_both.plot(true_traj[:, 0], true_traj[:, 1],
                          color=col, lw=1.5, alpha=0.9, label="true")
        ax_traj_both.plot(pred_traj[:, 0], pred_traj[:, 1],
                          color=col, lw=1.5, alpha=0.6,
                          linestyle="--", label="pred")

    for ax, title in zip(
        [ax_traj_pred, ax_traj_true, ax_traj_both],
        ["predicted trajectories", "true trajectories",
         "true (—) vs predicted (--)"]
    ):
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.4)

    plt.suptitle("Kinematic decoding results", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    return fig