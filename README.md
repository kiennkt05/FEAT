# FEAT

This repository implements **Federated gEometry-Aware correcTion (FEAT)** from
[Qi et al., *From Selection to Scheduling* (arXiv:2604.08617)](https://arxiv.org/pdf/2604.08617).
Select the implementation with `--implementation {repo,paper}` (default:
`paper`). `paper` is this project's **paper-guided FEAT_F reconstruction**: a ResNet-18 feature extractor and
fixed simplex ETF classifier, geometric structure alignment (GSA) after the
first task, energy-based geometric correction (EGC) at inference, and
FedCBDR-style exemplar replay. It is not a copy of the authors' published
trainer. The paper also evaluates **FEAT_R** with Re-Fed+
replay; that variant is not implemented here.

The replay selector extracts features with the trained global model, applies
orthogonal masking, pools pseudo-features for a global SVD, and uses leverage
scores for class-balanced sampling without replacement. It maps selected rows
back to local examples. This implementation uses a signed row permutation in
place of FedCBDR's dense row-orthogonal mask, so it is a practical adaptation.
FedCBDR's separate temperature-scaling loss is not part of this implementation.

`repo` is a compatibility port of the authors'
[published `methods/FEAT.py`](https://github.com/qizhuang-qz/FEAT/blob/main/methods/FEAT.py)
in `methods/FEAT_repo.py`. The published trainer is byte-identical to this
checkout's original `methods/FEAT.py` (Git blob
`2c08abe83a70d83b098a2cf84bb0cdea663824c0`). It keeps the original
first-task GSA loss, local
leverage-score replay with replacement, replay image composition, learning-rate
schedule, and best-model selection. The original repository did not include
`utils/inc_net.py`, so both paths use the currently available ETF network.
The restored path includes small import, device, worker, indexing, and
classifier-shape repairs required to run it. These two modes are distinct
implementations; their `--mem_size` budgets differ.
The original selector builds a dense square orthogonal mask per client, so
`repo` mode can require substantially more memory during exemplar selection.
The upstream trainer snapshots `best_model` before loading the current round's
aggregated weights, then restores that snapshot at task end; `repo` retains
this behavior, which can discard the last round's update.

## Setup

Run commands from the repository root with Python, PyTorch, torchvision,
NumPy, Pillow, and tqdm installed. Install a PyTorch/torchvision pair suitable
for your CPU or CUDA environment, then install the remaining packages:

```powershell
python -m pip install numpy pillow tqdm
```

The loader uses `../datasets/` relative to this directory. CIFAR-10 and
CIFAR-100 download through torchvision. `--dataset tiny_imagenet` downloads and
loads **the full 200-class Tiny ImageNet**, whereas the paper uses
**TinyImageNet-Subset**. The paper does not specify the subset's class IDs in
its experiment section, and this repository has no subset loader. Tiny ImageNet
runs here are exploratory and must not be compared directly with the paper's
TinyImageNet-Subset results. `--num_class` does not select a subset:
`main.py` sets it to 200 for `tiny_imagenet`.

## LANDER Exp5a-Global

`--method=exp5a_global` runs either FEAT training implementation with LANDER's
Exp5a-Global replay selector in place of its existing selector. The selector
accumulates exact sample contributions to the five trainable BatchNorm affine
updates in the final ResNet stage (FEAT's ETF classifier is fixed), validates
the SGD attribution, then uses LANDER's pooled uncentered PCA and client-wise
residual greedy selection. `--implementation=paper` keeps FEAT_F's loss and
replay behavior; `--implementation=repo` keeps the original repository's GSA
loss (`--kd`), learning rate schedule, replay composition (`--n`, `--tttt`),
and best-model snapshot behavior. The repo implementation stores selected
examples as its original `IndexedDataset` objects and selects memory only
before the final task, as in its original trainer.
The attribution gate checks the executed communication rounds before the repo
trainer restores its best-model snapshot; replay trajectories include those
rounds.

```powershell
python main.py --method=exp5a_global --dataset=cifar100 --tasks=5 --num_users=5 --beta=0.5 --gdr_task_budget=1000 --exp5a_rank=128
```

For the original repository training path:

```powershell
python main.py --method=exp5a_global --implementation=repo --dataset=cifar100 --tasks=5 --num_users=5 --beta=0.5 --gdr_task_budget=1000 --exp5a_rank=128 --kd=0.1
```

For this method, FEAT keeps its `--beta=0.5` default. Projection rank is `128`,
oversampling is `16`, and attribution tolerance is `0.05`. The total task replay
budget follows LANDER's dataset/task table unless `--gdr_task_budget` is given.
The FEAT data loader uses `../datasets/`; CPU and CUDA runs are supported.

## One FEAT_F run

This is the paper's CIFAR-100, 5-task, 5-client, `beta=0.1` setting with this
repository's default FEAT hyperparameters:

```powershell
python main.py --method feat --implementation paper --dataset cifar100 --tasks 5 --num_users 5 --beta 0.1 --com_round 100 --local_ep 2 --local_bs 128 --local_lr 0.04 --weight_decay 1e-5 --mem_size 50 --lambda 0.1 --temp 0.5 --ema_decay 0.7 --seed 2023
```

To run the original repository path with the same dataset and training
schedule, change to `--implementation repo`. In that path, `--kd` controls
GSA weight, while `--lambda` and `--temp` are unused:

```powershell
python main.py --method feat --implementation repo --dataset cifar100 --tasks 5 --num_users 5 --beta 0.1 --com_round 100 --local_ep 2 --local_bs 128 --local_lr 0.04 --weight_decay 1e-5 --mem_size 50 --kd 0.1 --ema_decay 0.7 --seed 2023
```

This example deliberately holds training settings constant for comparison.
The original CLI defaults were 2 communication rounds, 2 data-loader workers,
weight decay `5e-4`, and EMA decay `0.8`; supply those values explicitly if
you need the repository's original default configuration.

`--gpu` chooses the CUDA device when CUDA is available; CPU is also supported.
The runner prints the Top-1 accuracy after each task as `CNN top1 curve` and
writes a timestamped `feat_<implementation>_training_*.log` in the current
directory. In `paper` mode, add `--eval_each_round` to log intermediate round
accuracy; `repo` mode already evaluates every round. The current runner
does not save a CSV, plotted curve, checkpoint, or multi-seed summary. Its
`--wandb` option does not initialize Weights & Biases.

## Paper settings and replay budget

[Section 5.1 of the paper](https://arxiv.org/pdf/2604.08617) uses ResNet-18,
100 communication rounds per task, 2 local epochs, batch size 128, SGD learning
rate 0.04, and weight decay `1e-5`. For GSA it searches
`--lambda` in `{0.05, 0.1, 0.5}` and `--temp` in `{0.07, 0.5}`; for EGC it
searches `--ema_decay` in `{0.5, 0.7, 0.9}`. The example above uses
`0.1 / 0.5 / 0.7`; the paper does not identify one universal best combination.

`--mem_size` is an **integer per-class, per-client** budget in `paper` mode. For a
task with `C` new classes and `K` clients, the replay selector caps each client
at `C * mem_size` new exemplars and targets up to `K * C * mem_size` in total.
The cap may not be filled when client data are scarce. The paper reports a
per-task exemplar budget; the table maps it to this CLI's per-client cap, so
it should not be read as proof that the paper used identical memory accounting.
In `repo` mode, the original selector requests `C * mem_size` examples
**total across all clients** and samples with replacement.

| Dataset | Tasks | `--beta` | `--num_users` | Paper budget per task | CLI `--mem_size` and resulting per-client cap |
| --- | ---: | --- | --- | ---: | --- |
| CIFAR-10 | 3 | 0.5, 1.0 | 5, 10 | 450 | 150 gives 450, 450, **600** for the 3/3/4-class tasks |
| CIFAR-10 | 5 | 0.5, 1.0 | 5, 10 | 300 | 150 gives 300 for each 2-class task |
| CIFAR-100 | 5 | 0.1, 0.5 | 5, 10 | 1000 | 50 gives 1000 for each 20-class task |
| CIFAR-100 | 10 | 0.1, 0.5 | 5, 10 | 500 | 50 gives 500 for each 10-class task |
| TinyImageNet-Subset | 5 | 0.1, 0.5 | 5, 10 | 500 | Not reproducible with the current full-200-class loader |
| TinyImageNet-Subset | 10 | 0.1, 0.5 | 5, 10 | 250 | Not reproducible with the current full-200-class loader |

The 3-task CIFAR-10 split is 3/3/4 classes in this code; no single integer
`--mem_size` produces 450 for all three tasks. An exact 450-per-task cap would
require a per-task memory-budget option. The TinyImageNet-Subset rows require
its class list and matching loader before an exact `--mem_size` can be chosen.

## Table 1: main comparison

The paper reports mean and standard deviation over three random seeds for
each dataset/task/`beta`/client combination. Its seed IDs are not specified.
The following PowerShell sweep runs **FEAT_F on all 16 CIFAR configurations**
for three example seeds (48 runs). It does not run the paper's baselines,
FEAT_R, or TinyImageNet-Subset. Runs are sequential and can take substantial
time; each one contains 100 rounds **per task**.

```powershell
$common = @(
    "--method", "feat", "--implementation", "paper",
    "--com_round", "100", "--local_ep", "2",
    "--local_bs", "128", "--local_lr", "0.04", "--weight_decay", "1e-5",
    "--lambda", "0.1", "--temp", "0.5", "--ema_decay", "0.7"
)
$settings = @(
    @{ Dataset = "cifar10";  Tasks = @(3, 5);  Betas = @(0.5, 1.0); MemSize = 150 },
    @{ Dataset = "cifar100"; Tasks = @(5, 10); Betas = @(0.1, 0.5); MemSize = 50 }
)
$seeds = @(2023, 2024, 2025) # Example seeds; the paper does not name its seeds.

foreach ($setting in $settings) {
    foreach ($taskCount in $setting.Tasks) {
        foreach ($beta in $setting.Betas) {
            foreach ($clients in @(5, 10)) {
                foreach ($seed in $seeds) {
                    python main.py @common --dataset $setting.Dataset --tasks $taskCount `
                        --beta $beta --num_users $clients --mem_size $setting.MemSize `
                        --seed $seed
                    if ($LASTEXITCODE -ne 0) { throw "FEAT run failed" }
                }
            }
        }
    }
}
```

For an **exploratory full Tiny ImageNet** sweep, reuse `$common` and `$seeds`
from the preceding block:

```powershell
foreach ($taskCount in @(5, 10)) {
    foreach ($beta in @(0.1, 0.5)) {
        foreach ($clients in @(5, 10)) {
            foreach ($seed in $seeds) {
                python main.py @common --dataset tiny_imagenet --tasks $taskCount `
                    --beta $beta --num_users $clients --mem_size 12 --seed $seed
                if ($LASTEXITCODE -ne 0) { throw "FEAT run failed" }
            }
        }
    }
}
```

With 200 classes and 5 tasks, `--mem_size 12` caps each client at 480
exemplars per task; with 10 tasks it caps each client at 240. These are close
to the paper's 500/250 counts, but the dataset and budgets differ.

## Other paper experiments

The following commands reuse `$common` from the Table 1 sweep. Each example
uses one seed; repeat with three seeds and aggregate results for a comparison
to paper tables and figures.

### Table 2: ETF, GSA, and EGC ablation

The paper uses CIFAR-10/3 tasks with `beta=0.5,1.0` and CIFAR-100/5 tasks with
`beta=0.1,0.5`, all with 5 clients. The full FEAT_F row uses the Table 1
commands. Passing `--lambda 0` removes GSA's contribution and gives the
`+ETF+EGC` configuration:

```powershell
python main.py @common --dataset cifar10 --tasks 3 --num_users 5 --beta 0.5 --mem_size 150 --lambda 0 --seed 2023
```

There is currently no CLI switch to disable EGC or replace the ETF classifier.
Thus the FedCBDR, `+ETF`, and `+ETF+GSA` rows cannot be run exactly from this
repository. A zero GSA weight still computes the loss, but contributes no GSA
gradient.

### Figure 5 and Table 4: accuracy across tasks

Run the relevant Table 1 settings and use the printed `CNN top1 curve` and
`CNN: ...` grouped accuracies after each task. Figure 5 uses `beta=0.5` and
5 clients for CIFAR-10/5 tasks, CIFAR-100/10 tasks, and
TinyImageNet-Subset/5 tasks. For example:

```powershell
python main.py @common --dataset cifar100 --tasks 10 --num_users 5 --beta 0.5 --mem_size 50 --seed 2023 --eval_each_round
```

Table 4 evaluates the final model by task on CIFAR-10/3 tasks and
CIFAR-100/5 tasks, `beta=0.5`, 5 clients. The runner logs grouped accuracies
after each task, but does not export the paper's final per-task table; collect
the final `CNN: ...` line. Its missing ablation variants also limit Table 4.
The TinyImageNet-Subset curve remains unavailable without the subset loader.

### Figure 6: replay volume

For CIFAR-10/5 tasks, the paper's per-task budgets `{100, 150, 300}` map
exactly to this code's `--mem_size` values `{50, 75, 150}`. For example:

```powershell
foreach ($memory in @(50, 75, 150)) {
    python main.py @common --dataset cifar10 --tasks 5 --num_users 5 --beta 0.5 --mem_size $memory --seed 2023
    if ($LASTEXITCODE -ne 0) { throw "FEAT run failed" }
}
```

For CIFAR-100/5 tasks, the paper's `{125, 250, 500}` budgets would require
`--mem_size` values `{6.25, 12.5, 25}`. Only 500 is exact with the current
integer option; `{6, 12, 25}` yields `{120, 240, 500}` per client. The
TinyImageNet-Subset `{250, 500}` sweep needs the subset loader and an exact
per-task budget option if its class count does not divide these budgets.
To run the available CIFAR-100 approximations:

```powershell
foreach ($memory in @(6, 12, 25)) {
    python main.py @common --dataset cifar100 --tasks 5 --num_users 5 --beta 0.5 --mem_size $memory --seed 2023
    if ($LASTEXITCODE -ne 0) { throw "FEAT run failed" }
}
```

### Figure 7: GSA and EGC parameter sensitivity

Use CIFAR-10/5 tasks, `beta=0.5`, 5 clients. Vary one parameter at a time:

```powershell
$case = @("--dataset", "cifar10", "--tasks", "5", "--num_users", "5", "--beta", "0.5", "--mem_size", "150", "--seed", "2023")
foreach ($weight in @(0.05, 0.1, 0.5)) {
    python main.py @case --method feat --implementation paper --lambda $weight --temp 0.5 --ema_decay 0.7
}
foreach ($temperature in @(0.07, 0.5)) {
    python main.py @case --method feat --implementation paper --lambda 0.1 --temp $temperature --ema_decay 0.7
}
foreach ($decay in @(0.5, 0.7, 0.9)) {
    python main.py @case --method feat --implementation paper --lambda 0.1 --temp 0.5 --ema_decay $decay
}
```

These commands rely on the runner's paper-matching defaults for rounds,
epochs, batch size, learning rate, and weight decay. The paper is internally
inconsistent here: Section 5.7's paragraph lists `lambda={1,5,10}` and
`temp={0.1,0.5}`, while Figure 7's caption and Section 5.1 list the ranges
used above. The caption also ambiguously describes which parameters are held
fixed, so this is a one-parameter-at-a-time sweep rather than a guaranteed
exact reconstruction of Figure 7. Record the chosen values with every result.

### Figure 8: local epochs

Vary `--local_ep` over `{1, 2, 5, 10}` for each dataset/task/client setting
of interest. For example, on CIFAR-100/5 tasks:

```powershell
foreach ($epochs in @(1, 2, 5, 10)) {
    python main.py @common --dataset cifar100 --tasks 5 --num_users 5 --beta 0.1 --mem_size 50 --local_ep $epochs --seed 2023
    if ($LASTEXITCODE -ne 0) { throw "FEAT run failed" }
}
```

The figure also compares Re-Fed+, FedCBDR, and FEAT_R, which this runner does
not provide. The TinyImageNet-Subset series needs its loader.

### Table 3 and qualitative studies

Table 3 compares FedCBDR with MOON, FedRCL, CLIP2FL, FedFSA, and FEAT. Only
the FEAT_F configurations are available here. The paper's attention maps and
EGC case study require feature/attention export and plotting code that is not
in this runner. Section 5.5's communication analysis likewise has no
instrumentation here; the local federation simulator aggregates two EGC EMA
statistics alongside model parameters, but does not record bytes transferred.

## Quick verification

Run the small local tests before a long sweep:

```powershell
python -m unittest discover -s tests -q
```
