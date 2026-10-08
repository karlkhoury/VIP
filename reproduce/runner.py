"""
reproduce/runner.py
───────────────────
Writes the config of every experiment of the paper and runs it with
reproduce/run_experiment.py (run.py itself is not modified).

Setting: one channel only (AWGN by default, or Rayleigh / Rician via `channel`), used
for both training and testing, no channel-type input, J task tokens per task, total cap
K and per-task maximum K/2, two-stage training (frozen pipeline, then the allocator
alone), three seeds, fixed noise and fading shared by every method.

Runs are resumable: a seed whose per_sentence.csv.gz exists is skipped, so the
notebook can be re-run after a Colab disconnect. Outputs go to `out_root`
(a Google Drive folder on Colab).
"""

import copy
import os
import subprocess
import sys

import yaml

from common.channel import CHANNELS
from common.utils import deep_update

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNRS = [-15, -10, -5, -2, 0, 2, 5, 10, 15, 20]
EQUAL_W = [0.3333, 0.3333, 0.3334]
ABLATION = {   # name: (use_sentence, use_snr, use_priority, use_confidence, label)
    "full":          (True,  True,  True,  False, "CLS + SNR + priority"),
    "full_conf":     (True,  True,  True,  True,  "CLS + SNR + priority + confidence"),
    "snr_prio":      (False, True,  True,  False, "SNR + priority"),
    "snr_prio_conf": (False, True,  True,  True,  "SNR + priority + confidence"),
    "cls_snr":       (True,  True,  False, False, "CLS + SNR"),
    "snr":           (False, True,  False, False, "SNR only"),
    "cls_prio":      (True,  False, True,  False, "CLS + priority"),
    "cls":           (True,  False, False, False, "CLS only"),
    "prio":          (False, False, True,  False, "priority only"),
    "none":          (False, False, False, False, "no input"),
}


class Study:
    def __init__(self, out_root="runs/paper", seeds=(0, 1, 2), tokens_per_task=32,
                 caps=(8, 16, 32, 64), main_cap=16,
                 lambda_grid=(0.001, 0.002, 0.003, 0.005, 0.01, 0.02),
                 epsilon=0.25, snrs=SNRS, priority_snrs=(-10, -5, 0),
                 priority_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                 ablation=tuple(ABLATION), data_path="data/phrasebank_multitask.parquet",
                 smoke=False, channel="AWGN", tag="", alloc_epochs=30, val_draws=1):
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        self.channel = channel
        # stage-2 version: `tag` is appended to every allocator run name (the stage-1
        # pipeline is shared), so a new training recipe never reuses or overwrites old runs
        self.tag = tag
        self.alloc_epochs = alloc_epochs
        self.val_draws = val_draws
        self.out_root = out_root
        self.seeds = list(seeds)
        self.J = tokens_per_task
        self.caps = list(caps)
        self.main_cap = main_cap
        self.lambda_grid = [float(x) for x in lambda_grid]
        self.epsilon = epsilon
        self.snrs = list(snrs)
        self.priority_snrs = list(priority_snrs)
        self.priority_grid = list(priority_grid)
        self.ablation = list(ablation)
        self.data_path = data_path
        self.smoke = smoke
        self.config_dir = os.path.join(out_root, "configs")
        os.makedirs(self.config_dir, exist_ok=True)

    # ── names ────────────────────────────────────────────────────────────────
    PIPELINE = "pipeline"

    def alloc_name(self, cap, lam):
        return f"cap{cap}_lam{lam:g}{self.tag}"

    def ablation_name(self, variant, lam):
        return f"ablation_{variant}_lam{lam:g}{self.tag}"

    def priority_name(self, mode, lam):
        return f"priority_{mode}_lam{lam:g}{self.tag}"

    def run_dir(self, name):
        return os.path.join(self.out_root, name)

    # ── configs ──────────────────────────────────────────────────────────────
    def _base(self):
        with open(os.path.join(REPO, "runs", "configs", "base_tokenalloc.yaml")) as f:
            cfg = yaml.safe_load(f)
        cfg = deep_update(cfg, {
            "seeds": self.seeds, "noise_seed": 1234, "tokens_per_task": self.J,
            "data": {"path": self.data_path},
            "allocator": {"use_channel": False},
            "train": {"channels": [self.channel]},
            "eval": {"channels": [self.channel], "snrs": self.snrs, "priorities": [EQUAL_W],
                     "saturation": False},
        })
        if self.smoke:   # tiny random encoder + synthetic data: checks the code path on CPU
            cfg = deep_update(cfg, {
                "data": {"path": os.path.join(self.out_root, "synthetic.parquet"), "synthetic": True,
                         "n": 96, "fractions": [0.6, 0.2, 0.2]},
                "encoder": {"name": "tiny-random", "hidden": 64, "unfreeze_last_n": -1},
                "train": {"batch_size": 8, "max_steps_per_epoch": 3},
            })
        return cfg

    def _write(self, name, cfg):
        cfg = copy.deepcopy(cfg)
        cfg["name"] = name
        cfg["out_dir"] = self.run_dir(name)
        path = os.path.join(self.config_dir, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return path

    def pipeline_config(self):
        cfg = deep_update(self._base(), {
            "cap": 3 * self.J, "save_checkpoint": True,
            "train": {"epochs_warmup": 1 if self.smoke else 12, "epochs_allocator": 0},
            "eval": {"policies": ["all"], "saturation": True, "splits": ["test"]},
        })
        return self._write(self.PIPELINE, cfg)

    def _alloc(self, cap, lam, allocator=None, priorities=None, snrs=None, splits=("test",),
               policies=("allocator", "equal", "all")):
        return deep_update(self._base(), {
            "cap": cap, "max_tokens_per_task": min(self.J, max(1, cap // 2)), "save_checkpoint": True,
            "init_checkpoint": os.path.join(self.run_dir(self.PIPELINE), "seed{seed}", "model.pt"),
            "allocator": allocator or {},
            "train": {"allocator_stage": "frozen", "epochs_allocator": 2 if self.smoke else self.alloc_epochs,
                      "lr_allocator": 1.0e-3, "lambda": float(lam), "val_draws": self.val_draws},
            "eval": {"policies": list(policies), "splits": list(splits),
                     "priorities": priorities or [EQUAL_W], **({"snrs": list(snrs)} if snrs else {})},
        })

    def lambda_config(self, lam, cap=None):
        """Allocator at the main cap for one price; evaluated on validation AND test."""
        cap = cap or self.main_cap
        return self._write(self.alloc_name(cap, lam), self._alloc(cap, lam, splits=("val", "test")))

    def cap_config(self, cap, lam):
        return self._write(self.alloc_name(cap, lam), self._alloc(cap, lam, splits=("val", "test")))

    def _three_priorities(self):
        return [EQUAL_W, [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]

    def ablation_config(self, variant, lam):
        s, n, p, c, _ = ABLATION[variant]
        alloc = {"use_sentence": s, "use_snr": n, "use_priority": p, "use_confidence": c}
        cfg = self._alloc(self.main_cap, lam, allocator=alloc, priorities=self._three_priorities(),
                          policies=("allocator", "equal"))
        if variant == "full":   # the selected allocator itself, re-evaluated (no retraining)
            cfg["load_allocator"] = os.path.join(self.run_dir(self.alloc_name(self.main_cap, lam)),
                                                 "seed{seed}", "allocator.pt")
        return self._write(self.ablation_name(variant, lam), cfg)

    def priority_config(self, mode, lam):
        """Priority sweep: 'gain' re-evaluates the selected allocator, 'embed' trains the
        figure-style allocator (priority embedded in the hidden layer) and sweeps it."""
        sweep = [[round((1 - w) / 2, 4), round((1 - w) / 2, 4), w] for w in self.priority_grid]
        cfg = self._alloc(self.main_cap, lam, allocator={"priority_mode": mode}, priorities=sweep,
                          snrs=self.priority_snrs, policies=("allocator", "equal"))
        if mode == "gain":
            cfg["load_allocator"] = os.path.join(self.run_dir(self.alloc_name(self.main_cap, lam)),
                                                 "seed{seed}", "allocator.pt")
        return self._write(self.priority_name(mode, lam), cfg)

    # ── running ──────────────────────────────────────────────────────────────
    def done_seeds(self, name):
        return [s for s in self.seeds
                if os.path.exists(os.path.join(self.run_dir(name), f"seed{s}", "per_sentence.csv.gz"))]

    def run(self, config_path, quiet=True):
        """Run the missing seeds of one experiment; print progress lines only."""
        name = os.path.splitext(os.path.basename(config_path))[0]
        todo = [s for s in self.seeds if s not in self.done_seeds(name)]
        if not todo:
            print(f"[skip] {name}: all seeds done")
            return
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                   HF_HUB_DISABLE_SYMLINKS_WARNING="1")
        cmd = [sys.executable, "-m", "reproduce.run_experiment", "--config", config_path,
               "--seeds", *map(str, todo)]
        if self.smoke:
            cmd += ["--device", "cpu"]
        os.makedirs(self.run_dir(name), exist_ok=True)
        log = os.path.join(self.run_dir(name), "run.log")
        keep = ("seed ", "[warmup", "[allocator", "attention-rule", "Error", "error", "Traceback")
        print(f"[run] {name}: seeds {todo}")
        with open(log, "a", encoding="utf-8") as f, subprocess.Popen(
                cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace") as p:
            for line in p.stdout:
                f.write(line)
                if not quiet or any(k in line for k in keep):
                    print("   " + line.rstrip())
        if p.returncode != 0:
            raise RuntimeError(f"{name} failed (exit {p.returncode}); see {log}")


SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CAP8_VARIANTS = {   # name: (allocator settings, label)
    "gain":  ({}, "Learned allocator"),
    "embed": ({"priority_mode": "embed"}, "Learned allocator, priority embedding"),
}


class SmallStudy(Study):
    """
    Small token budget (Part A of the notebook): 4 task tokens per task, one channel,
    no channel-type input, per-task maximum = 4.

      * pipeline: stage 1, nested dropout, saturation curves for k = 1..4
      * fixed cap 8, lambda = 0: allocator (gain form, and the priority-embedding variant)
        against the equal, proportional and random splits and the hindsight oracle
      * variable rate at cap 12: a sweep of the price lambda (one allocator per lambda)
      * priority: ESG priority swept at 0 dB on the cap-8 allocators (gain vs embedding),
        and one allocator trained only at w = (0.1, 0.1, 0.8)
      * optional: the allocator trained jointly with the pipeline (one stage)
    """
    PIPELINE = "j4_pipeline"

    def __init__(self, out_root="runs/paper", seeds=(0, 1, 2), channel="AWGN", fixed_cap=8, rate_cap=12,
                 rate_lambdas=(0.001, 0.003, 0.01, 0.03, 0.1), snrs=SNRS, priority_snrs=(0,),
                 priority_grid=SWEEP, data_path="data/phrasebank_multitask.parquet", smoke=False,
                 tag="", alloc_epochs=30, val_draws=1):
        super().__init__(out_root=out_root, seeds=seeds, tokens_per_task=4, caps=(fixed_cap,), main_cap=fixed_cap,
                         lambda_grid=rate_lambdas, snrs=snrs, priority_snrs=priority_snrs,
                         priority_grid=priority_grid, ablation=(), data_path=data_path, smoke=smoke,
                         channel=channel, tag=tag, alloc_epochs=alloc_epochs, val_draws=val_draws)
        self.fixed_cap, self.rate_cap = fixed_cap, rate_cap

    def cap8_name(self, variant):
        return f"j4_cap8_{variant}{self.tag}"

    def rate_name(self, lam, conf=False):
        return f"j4_cap12_lam{lam:g}" + ("_conf" if conf else "") + self.tag

    def priority_name(self, mode, lam=None):
        return f"j4_priority_{mode}{self.tag}"

    @property
    def FIXED_PRIORITY(self):
        return f"j4_fixed_priority{self.tag}"

    JOINT = "j4_joint_cap8"

    def _allocator_path(self, name):
        return os.path.join(self.run_dir(name), "seed{seed}", "allocator.pt")

    def cap8_config(self, variant):
        cfg = self._alloc(self.fixed_cap, 0.0, allocator=CAP8_VARIANTS[variant][0],
                          policies=("allocator", "equal", "proportional", "random_matched", "oracle_greedy"))
        return self._write(self.cap8_name(variant), cfg)

    def rate_config(self, lam, conf=False):
        cfg = self._alloc(self.rate_cap, lam, allocator={"use_confidence": True} if conf else {},
                          policies=("allocator", "equal", "fixed:4-3-3", "fixed:3-3-2"))
        return self._write(self.rate_name(lam, conf), cfg)

    def priority_config(self, mode, lam=None):
        """Re-evaluates the cap-8 allocator (gain or embed) over the ESG priority sweep."""
        variant = "gain" if mode == "gain" else "embed"
        sweep = [[round((1 - w) / 2, 4), round((1 - w) / 2, 4), w] for w in self.priority_grid]
        cfg = self._alloc(self.fixed_cap, 0.0, allocator=CAP8_VARIANTS[variant][0], priorities=sweep,
                          snrs=self.priority_snrs, policies=("allocator", "equal", "proportional", "oracle_greedy"))
        cfg["load_allocator"] = self._allocator_path(self.cap8_name(variant))
        return self._write(self.priority_name(mode), cfg)

    def fixed_priority_config(self, w=(0.1, 0.1, 0.8)):
        """One allocator trained (and validated) only at priority w, tested at w."""
        cfg = self._alloc(self.fixed_cap, 0.0, priorities=[list(w)], snrs=self.priority_snrs,
                          policies=("allocator", "equal", "proportional", "oracle_greedy"))
        cfg["train"]["priority"] = {"mode": "fixed", "w": list(w)}
        return self._write(self.FIXED_PRIORITY, cfg)

    def joint_config(self):
        """Allocator trained together with the pipeline in one stage (the approach we dropped)."""
        cfg = deep_update(self._base(), {
            "cap": self.fixed_cap, "save_checkpoint": False,
            "train": {"allocator_stage": "joint", "epochs_warmup": 1 if self.smoke else 4,
                      "epochs_allocator": 1 if self.smoke else 8, "lambda": 0.0},
            "eval": {"policies": ["allocator", "equal"], "splits": ["test"]},
        })
        return self._write(self.JOINT, cfg)
