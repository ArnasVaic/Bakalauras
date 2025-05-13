# %%
import numpy as np
from solver_utils import show_solution_frames, pretty_time
from solvers.adi.time_step_strategy import SCGQMStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config
from solvers.adi.solver import Solver
import matplotlib.pyplot as plt
from tqdm import tqdm

ORDER = 3
T = 1000
OPTIMAL_MIX_TIME = 2/3 * 3600
SAMPLES = 20

# %% Baseline
config = large_config(order=ORDER, temperature=T)
config.frame_stride = 100
c0 = initial_condition(config)
t, c = Solver(config).solve(c0, lambda f: f.copy())
baseline = np.array([t[-1]])
np.save(f'paper/large-random-mix/baseline-T{T}-ord{ORDER}.npy', baseline)

# %% Solve

durations = np.zeros(SAMPLES)

for sample in tqdm(range(SAMPLES)):
  mix_config = MixConfig('random', [ OPTIMAL_MIX_TIME ])
  config = large_config(order=ORDER, temperature=T, mix_config=mix_config)
  config.frame_stride = 1000000 # we dont really need any of the frames
  c0 = initial_condition(config)
  t, c = Solver(config).solve(c0, lambda _: 0)
  durations[sample] = t[-1]

np.save(f'paper/large-random-mix/sample-durations-ord{ORDER}-T{T}.npy', durations)
# %% Show

# Set global font sizes
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)   # figure title size

# Load data
durations = np.load('paper/large-random-mix/sample-durations.npy')  # in seconds
baseline = np.load('paper/large-random-mix/baseline-T1000-ord01.npy')

# Number of samples
SAMPLES = len(durations)
xs = np.arange(SAMPLES)

# Convert seconds to hours
durations_hours = durations / 3600
baseline_hours = baseline / 3600
avg_hours = np.full(SAMPLES, np.mean(durations_hours))

# Plotting
plt.plot(xs, np.repeat(baseline_hours, SAMPLES), label='Trukmė nemaišant', color='orange', linestyle='dashed',)
plt.plot(xs, durations_hours, label='Atsitiktinis maišymas')
plt.plot(xs, avg_hours, linestyle='dashed', label='Vidutinė reakcijos trukmė', color='green')

# Labels and legend
plt.xlabel('Bandymo indeksas')
plt.ylabel('Reakcijos trukmė [val]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../paper/images/mixing/sample-durations-random-mix-ord0-T1000.png', dpi=300, bbox_inches='tight')
plt.show()

# %% One graph for all orders
import numpy as np
import matplotlib.pyplot as plt

# Set global font sizes (slightly increased)
plt.rc('font', size=16)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=24)

SAMPLES = 20
xs = np.arange(SAMPLES)

# Plot setup: 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

# Ensure axes is iterable
axes = np.ravel(axes)

# Keep handles and labels for global legend
handles, labels = [], []

# Loop over ord values 1–3 and corresponding subplot axes
for ord_val, ax in zip(range(1, 4), axes):
    # Convert to hours
    durations = np.load(f'paper/large-random-mix/sample-durations-ord{ord_val}-T1000.npy')
    durations_hours = durations / 3600
    avg_hours = np.mean(durations_hours)

    # Load baseline (same for all)
    baseline = np.load('paper/large-random-mix/baseline-T1000-ord1.npy')
    baseline_hours = baseline / 3600

    # Plot
    l1, = ax.plot(xs, durations_hours, label='Reakcijos trukmė modeliuojant atsitiktinį maišymą')
    l2 = ax.hlines(baseline_hours, xmin=0, xmax=SAMPLES - 1, colors='orange', linestyles='solid', label='Bazinė reakcijos trukmė')
    l3 = ax.hlines(avg_hours, xmin=0, xmax=SAMPLES - 1, colors='green', linestyles='dashed', label='Vidutinė reakcijos trukmė maišant')

    if ord_val == 1:
        handles.extend([l1, l2, l3])

    ax.set_xlim((0, 19))          # prevents limits padding

    ax.set_title(f'Erdvė padidinta {4 ** ord_val} kartus')
    ax.grid(True)

# Only one shared X and Y label
fig.text(0.5, 0.02, 'Bandymo indeksas', ha='center', fontsize=18)
fig.text(0.07, 0.5, 'Reakcijos trukmė [val]', va='center', rotation='vertical', fontsize=18)

# Add a common legend above plots
fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.12))

# Remove all padding
plt.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14, wspace=0.1)
fig.savefig('../paper/images/mixing/sample-durations-random-mix.png', dpi=300, bbox_inches='tight')
plt.show()


