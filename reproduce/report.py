"""
reproduce/report.py
───────────────────
Tables, figures and paired-bootstrap significance tests for the paper notebook.

Every number is computed from the per-sentence logs written by run_experiment.py.
A "case" is one (seed, SNR, priority, sentence) on the fixed shared noise; two
methods are compared on the same cases (paired bootstrap, 10,000 resamples).
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from common.stats import paired_bootstrap


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


NOTEBOOK = _in_notebook()
import matplotlib  # noqa: E402
if not NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEY = ["seed", "channel", "snr_db", "sentence_id", "w_S", "w_F", "w_E"]
COLOR = {"all": "#7a7a7a", "equal": "#2b2b2b", "allocator": "#2a78d6", "embed": "#eb6834",
         "S": "#2a78d6", "F": "#1baf7a", "E": "#eb6834"}
LABEL = {"all": "No allocator, no cap (all tokens)", "equal": "No allocator, equal split at cap",
         "allocator": "Learned allocator"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def md(text):
    if NOTEBOOK:
        from IPython.display import Markdown, display
        display(Markdown(text))
    else:
        print(text)


def table(df, title=None, digits=2):
    if title:
        md(f"**{title}**")
    if NOTEBOOK:
        from IPython.display import display
        num = df.select_dtypes("number").columns
        display(df.style.format({c: f"{{:.{digits}f}}" for c in num}, na_rep="").hide(axis="index"))
    else:
        print(df.to_string(index=False, float_format=lambda x: f"{x:.{digits}f}"))


def ci(r, digits=2):
    return f"{r['delta']:+.{digits}f} [{r['lo']:+.{digits}f}, {r['hi']:+.{digits}f}]"


class Report:
    def __init__(self, study):
        self.s = study
        self.fig_dir = os.path.join(study.out_root, "report")
        os.makedirs(self.fig_dir, exist_ok=True)
        self.lam = None
        self.findings = []

    # ── loading ──────────────────────────────────────────────────────────────
    def load(self, name, split="test", lam=0.0):
        files = sorted(glob.glob(os.path.join(self.s.run_dir(name), "seed*", "per_sentence.csv.gz")))
        if not files:
            raise FileNotFoundError(f"no results for {name}: run its cell first")
        d = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        d = d[d["split"] == split].copy()
        for c in ("w_S", "w_F", "w_E"):
            d[c] = d[c].round(2)
        d["tokens"] = d[["k_S", "k_F", "k_E"]].sum(axis=1)
        d["acc"] = d[["correct_S", "correct_F", "correct_E"]].mean(axis=1) * 100
        wsum = d[["w_S", "w_F", "w_E"]].sum(axis=1)
        d["wacc"] = (d.w_S * d.correct_S + d.w_F * d.correct_F + d.w_E * d.correct_E) / wsum * 100
        d["obj"] = (d.w_S * d.ce_S + d.w_F * d.ce_F + d.w_E * d.ce_E) / wsum + lam * d["tokens"]
        return d

    @staticmethod
    def paired(a, b, metric, n=10_000):
        m = a[KEY + [metric]].merge(b[KEY + [metric]], on=KEY, suffixes=("_a", "_b"))
        return paired_bootstrap(m[f"{metric}_a"].values, m[f"{metric}_b"].values, n)

    def save(self, fig, name):
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        for ax in fig.axes:                       # plain numbers on log axes (e.g. 64, not 6.4x10^1)
            for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
                if scale == "log":
                    axis.set_major_formatter(ScalarFormatter())
                    axis.set_minor_formatter(NullFormatter())
        path = os.path.join(self.fig_dir, f"{name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        if NOTEBOOK:
            plt.show()
        plt.close(fig)
        return path

    def note(self, text):
        self.findings.append(text)

    # ── 0. data ──────────────────────────────────────────────────────────────
    def dataset(self):
        from common.data import NUM_CLASSES, PhraseBankDataset, split_indices, template_groups
        cfg = json.load(open(os.path.join(self.s.run_dir(self.s.PIPELINE), "config.json")))
        ds = PhraseBankDataset(cfg["data"]["path"])
        y = ds.labels
        sp = split_indices(len(ds), tuple(cfg["data"].get("fractions", (0.7, 0.1, 0.2))),
                           cfg["data"].get("split_seed", 0), template_groups(ds.texts),
                           y[:, 0] * 12 + y[:, 1] * 4 + y[:, 2])
        names = {"sentiment": ["negative", "neutral", "positive"],
                 "fls": ["not FLS", "non-specific", "specific"],
                 "esg": ["environmental", "social", "governance", "none"]}
        rows = []
        for i, t in enumerate(("sentiment", "fls", "esg")):
            for c in range(NUM_CLASSES[t]):
                row = {"task": t, "class": names[t][c]}
                for s in ("train", "val", "test"):
                    row[s] = int((y[sp[s], i] == c).sum())
                row["total"] = int((y[:, i] == c).sum())
                rows.append(row)
        md(f"{len(ds):,} sentences: train {len(sp['train']):,} / validation {len(sp['val']):,} / "
           f"test {len(sp['test']):,} (stratified, near-duplicate templates kept in one split).")
        table(pd.DataFrame(rows), "Class counts per split", digits=0)

    # ── 1. saturation ────────────────────────────────────────────────────────
    def saturation(self):
        d = self.load(self.s.PIPELINE)
        u = d[d.method.str.startswith("uniform_k")].copy()
        u["k"] = u.method.str[9:].astype(int)
        acc = u.pivot_table(index="k", columns="snr_db", values="acc")
        acc["all SNRs"] = u.groupby("k").acc.mean()
        ks = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if k in acc.index]
        t = acc.loc[ks].reset_index()
        t.insert(1, "symbols", t.k * 12)
        t.columns = [c if isinstance(c, str) else f"{c:g} dB" for c in t.columns]
        t = t.rename(columns={"k": "tokens per task"})
        table(t, "No allocator: accuracy (%) when every task sends k task tokens")
        # tokens per task needed to come within 0.5 points of the best, per SNR
        need = {}
        for s in acc.columns.drop("all SNRs"):
            best = acc[s].max()
            need[f"{s:g} dB"] = int(acc.index[acc[s] >= best - 0.5].min())
        table(pd.DataFrame([need]), "Tokens per task needed to be within 0.5 points of the best accuracy", 0)
        lo, hi = min(need.values()), max(need.values())
        self.note(f"Token saturation: the tokens per task needed to reach within 0.5 points of the best "
                  f"accuracy range from {hi} at the lowest SNR to {lo} at high SNR, so a fixed budget "
                  f"either wastes symbols on clean channels or starves noisy ones.")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        snr_pick = [s for s in (-15, -10, -5, 0, 5, 20) if s in acc.columns]
        cmap = plt.get_cmap("viridis")
        for i, s in enumerate(snr_pick):
            ax[0].plot(acc.index * 12, acc[s], marker="o", ms=3, color=cmap(i / max(1, len(snr_pick) - 1)),
                       label=f"{s:g} dB")
        ax[0].set_xscale("log"); ax[0].set_xlabel("symbols per sentence (3 tasks x k tokens x 4)")
        ax[0].set_ylabel("mean accuracy (%)"); ax[0].set_title("No allocator: accuracy vs tokens, per SNR")
        ax[0].legend(fontsize=8)
        for key in "SFE":
            ax[1].plot(acc.index * 12, u.groupby("k")[f"correct_{key}"].mean() * 100, marker="o", ms=3,
                       color=COLOR[key], label={"S": "Sentiment", "F": "FLS", "E": "ESG"}[key])
        ax[1].set_xscale("log"); ax[1].set_xlabel("symbols per sentence"); ax[1].set_ylabel("accuracy (%)")
        ax[1].set_title("Per task, averaged over SNR"); ax[1].legend(fontsize=8)
        self.save(fig, "1_saturation")

    # ── 2. lambda selection (validation only) ────────────────────────────────
    def select_lambda(self, epsilon=None):
        eps = self.s.epsilon if epsilon is None else epsilon
        rows, eq_acc = [], []
        for lam in self.s.lambda_grid:
            d = self.load(self.s.alloc_name(self.s.main_cap, lam), "val", lam)
            a, e = d[d.method == "allocator"], d[d.method == "equal"]
            eq_acc.append(e.acc.mean())
            rows.append({"lambda": lam, "symbols": a.symbols.mean(), "accuracy": a.acc.mean(),
                         "vs equal split": a.acc.mean() - e.acc.mean(), "eq_sym": e.symbols.mean()})
        t = pd.DataFrame(rows)
        t["feasible"] = t["vs equal split"] >= -eps
        feas = t[t.feasible]
        chosen = (feas.sort_values(["symbols", "lambda"]).iloc[0] if len(feas)
                  else t.sort_values("lambda").iloc[0])
        self.lam = float(chosen["lambda"])
        show = t.drop(columns="eq_sym").copy()
        show["feasible"] = show.feasible.map({True: "yes", False: "no"})
        show["selected"] = np.where(show["lambda"] == self.lam, "<==", "")
        md(f"**Rule:** among the prices whose **validation** accuracy is within ε = {eps} points of the "
           f"equal split at cap {self.s.main_cap} ({np.mean(eq_acc):.2f}%, {t.eq_sym.iloc[0]:.0f} symbols), "
           f"choose the one that sends the fewest symbols. The test set is not used for this choice.")
        table(show, f"Validation results at cap {self.s.main_cap} (per-task max {self.s.main_cap // 2})", 3)
        md(f"**Selected λ = {self.lam:g}** (validation: {chosen.symbols:.1f} symbols, "
           f"{chosen.accuracy:.2f}% accuracy, {chosen['vs equal split']:+.2f} points vs the equal split).")
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(t.symbols, t.accuracy, marker="o", color=COLOR["allocator"], label="learned allocator (each λ)")
        for _, r in t.iterrows():
            ax.annotate(f"λ={r['lambda']:g}", (r.symbols, r.accuracy), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
        ax.scatter([t.eq_sym.iloc[0]], [np.mean(eq_acc)], color=COLOR["equal"], zorder=3,
                   label=f"equal split, cap {self.s.main_cap}")
        ax.axhline(np.mean(eq_acc) - eps, ls="--", color="#c0392b", lw=1, label=f"equal split − ε ({eps})")
        ax.scatter([chosen.symbols], [chosen.accuracy], s=180, facecolors="none", edgecolors="#c0392b",
                   linewidths=2, zorder=4, label=f"selected λ = {self.lam:g}")
        ax.set_xlabel("average symbols per sentence (validation)"); ax.set_ylabel("mean accuracy (%)")
        ax.set_title("Choosing λ on the validation set"); ax.legend(fontsize=8)
        self.save(fig, "2_lambda_selection")
        return self.lam

    # ── 3. the three benchmarks ──────────────────────────────────────────────
    def benchmarks(self):
        lam, cap = self.lam, self.s.main_cap
        d = self.load(self.s.alloc_name(cap, lam), "test", lam)
        g = {m: d[d.method == m] for m in ("all", "equal", "allocator")}
        names = {"all": f"No allocator, no cap ({self.s.J * 3} tokens)",
                 "equal": f"No allocator, cap {cap} (equal split)",
                 "allocator": f"Learned allocator, cap {cap}, λ = {lam:g}"}
        rows = []
        for m, x in g.items():
            rows.append({"method": names[m], "task tokens": x.tokens.mean(), "symbols": x.symbols.mean(),
                         "Sentiment": x.correct_S.mean() * 100, "FLS": x.correct_F.mean() * 100,
                         "ESG": x.correct_E.mean() * 100, "mean accuracy": x.acc.mean(),
                         "symbols saved vs equal (%)": (1 - x.symbols.mean() / g["equal"].symbols.mean()) * 100,
                         "symbols saved vs all (%)": (1 - x.symbols.mean() / g["all"].symbols.mean()) * 100})
        table(pd.DataFrame(rows), "The three benchmarks (test set, mean of seeds and SNRs)")
        eps = self.s.epsilon
        r_eq = self.paired(g["allocator"], g["equal"], "acc")
        r_eqs = self.paired(g["allocator"], g["equal"], "symbols")
        r_all = self.paired(g["allocator"], g["all"], "acc")
        r_alls = self.paired(g["allocator"], g["all"], "symbols")
        sig = pd.DataFrame([
            {"comparison": "allocator − equal split", "Δ accuracy [95% CI]": ci(r_eq), "Δ symbols [95% CI]": ci(r_eqs),
             f"non-inferior (CI low > −{eps})": "yes" if r_eq["lo"] > -eps else "no",
             "fewer symbols (significant)": "yes" if r_eqs["hi"] < 0 else "no"},
            {"comparison": "allocator − all tokens", "Δ accuracy [95% CI]": ci(r_all), "Δ symbols [95% CI]": ci(r_alls),
             f"non-inferior (CI low > −{eps})": "yes" if r_all["lo"] > -eps else "no",
             "fewer symbols (significant)": "yes" if r_alls["hi"] < 0 else "no"}])
        table(sig, f"Paired bootstrap (10,000 resamples, {r_eq['n_units']:,} paired cases)")
        save_eq = (1 - g["allocator"].symbols.mean() / g["equal"].symbols.mean()) * 100
        self.note(f"Main result (cap {cap}, λ = {lam:g}): the allocator sends {save_eq:.0f}% fewer symbols than "
                  f"the equal split ({g['allocator'].symbols.mean():.1f} vs {g['equal'].symbols.mean():.0f}); "
                  f"accuracy difference {ci(r_eq)} points"
                  + (f", non-inferior at ε = {eps}." if r_eq["lo"] > -eps else f", NOT non-inferior at ε = {eps}."))
        self.note(f"Against sending every token ({g['all'].symbols.mean():.0f} symbols), the allocator uses "
                  f"{(1 - g['allocator'].symbols.mean() / g['all'].symbols.mean()) * 100:.0f}% fewer symbols for "
                  f"{ci(r_all)} points of accuracy.")
        per = pd.DataFrame({f"{m}: symbols": x.groupby("snr_db").symbols.mean() for m, x in g.items()})
        for m, x in g.items():
            per[f"{m}: accuracy"] = x.groupby("snr_db").acc.mean()
        per = per.reset_index().rename(columns={"snr_db": "SNR (dB)"})
        table(per, "Per SNR (test)", 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for m, x in g.items():
            s = x.groupby("snr_db")
            ax[0].plot(s.symbols.mean().index, s.symbols.mean().values, marker="o", color=COLOR[m], label=names[m])
            ax[1].plot(s.acc.mean().index, s.acc.mean().values, marker="o", color=COLOR[m], label=names[m])
        ax[0].set_yscale("log"); ax[0].set_ylabel("symbols per sentence (log)"); ax[0].set_title("Bandwidth used")
        ax[1].set_ylabel("mean accuracy (%)"); ax[1].set_title("Accuracy")
        for a in ax:
            a.set_xlabel("SNR (dB)")
        ax[1].legend(fontsize=8)
        self.save(fig, "3_benchmarks_per_snr")

    # ── 4. trade-off curve over lambda (test) ────────────────────────────────
    def tradeoff(self):
        rows = []
        for lam in self.s.lambda_grid:
            d = self.load(self.s.alloc_name(self.s.main_cap, lam), "test", lam)
            a = d[d.method == "allocator"]
            rows.append({"lambda": lam, "symbols": a.symbols.mean(), "accuracy": a.acc.mean()})
        t = pd.DataFrame(rows)
        table(t, f"Test accuracy vs symbols for every λ (cap {self.s.main_cap}; λ was selected on validation)", 3)
        p = self.load(self.s.PIPELINE)
        u = p[p.method.str.startswith("uniform_k")].copy()
        u["k"] = u.method.str[9:].astype(int)
        c = u.groupby("k").agg(symbols=("symbols", "mean"), acc=("acc", "mean"))
        d = self.load(self.s.alloc_name(self.s.main_cap, self.lam), "test")
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        ax.plot(c.symbols, c.acc, marker=".", color=COLOR["all"], label="no allocator (k tokens per task)")
        ax.plot(t.symbols, t.accuracy, marker="o", color=COLOR["allocator"], label=f"learned allocator, cap {self.s.main_cap}")
        e = d[d.method == "equal"]
        ax.scatter([e.symbols.mean()], [e.acc.mean()], color=COLOR["equal"], zorder=3, s=50,
                   label=f"equal split, cap {self.s.main_cap}")
        sel = t[t["lambda"] == self.lam].iloc[0]
        ax.scatter([sel.symbols], [sel.accuracy], s=180, facecolors="none", edgecolors="#c0392b", linewidths=2,
                   zorder=4, label=f"selected λ = {self.lam:g}")
        ax.set_xscale("log"); ax.set_xlabel("average symbols per sentence (log)"); ax.set_ylabel("mean accuracy (%)")
        ax.set_title("Accuracy vs bandwidth (test)"); ax.legend(fontsize=8)
        self.save(fig, "4_tradeoff")

    # ── 5. caps ──────────────────────────────────────────────────────────────
    def caps(self):
        lam, rows = self.lam, []
        for cap in self.s.caps:
            d = self.load(self.s.alloc_name(cap, lam), "test", lam)
            a, e = d[d.method == "allocator"], d[d.method == "equal"]
            r, rs = self.paired(a, e, "acc"), self.paired(a, e, "symbols")
            rows.append({"cap": cap, "per-task max": cap // 2, "equal: symbols": e.symbols.mean(),
                         "equal: accuracy": e.acc.mean(), "allocator: symbols": a.symbols.mean(),
                         "allocator: accuracy": a.acc.mean(),
                         "symbols saved (%)": (1 - a.symbols.mean() / e.symbols.mean()) * 100,
                         "Δ accuracy [95% CI]": ci(r), "Δ symbols significant": "yes" if rs["hi"] < 0 else "no"})
        t = pd.DataFrame(rows)
        table(t, f"Every cap, λ = {lam:g} (test)")
        self.note("Across caps " + ", ".join(f"{int(r.cap)} ({r['symbols saved (%)']:.0f}% fewer symbols, "
                                             f"Δ acc {r['Δ accuracy [95% CI]']})" for _, r in t.iterrows()) + ".")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        x, w = np.arange(len(t)), 0.38
        ax[0].bar(x - w / 2, t["equal: symbols"], w, color=COLOR["equal"], label="no allocator, equal split")
        ax[0].bar(x + w / 2, t["allocator: symbols"], w, color=COLOR["allocator"], label=f"learned allocator, λ={lam:g}")
        ax[1].bar(x - w / 2, t["equal: accuracy"], w, color=COLOR["equal"])
        ax[1].bar(x + w / 2, t["allocator: accuracy"], w, color=COLOR["allocator"])
        for a, lab in ((ax[0], "symbols per sentence"), (ax[1], "mean accuracy (%)")):
            a.set_xticks(x); a.set_xticklabels([f"cap {c}\n(max {c // 2}/task)" for c in t.cap]); a.set_ylabel(lab)
        lo = min(t["equal: accuracy"].min(), t["allocator: accuracy"].min())
        ax[1].set_ylim(lo - 2, max(t["equal: accuracy"].max(), t["allocator: accuracy"].max()) + 1)
        ax[0].legend(fontsize=8); ax[0].set_title("Bandwidth"); ax[1].set_title("Accuracy")
        self.save(fig, "5_caps")

    # ── 6. input ablation ────────────────────────────────────────────────────
    def ablation(self):
        from reproduce.runner import ABLATION
        lam = self.lam
        frames = {v: self.load(self.s.ablation_name(v, lam), "test", lam) for v in self.s.ablation}
        full = frames["full"][frames["full"].method == "allocator"]
        eq = frames["full"][frames["full"].method == "equal"]
        lo_snr, hi_snr = min(self.s.snrs), max(self.s.snrs)
        rows = []
        for v, d in list(frames.items()) + [("equal", None)]:
            a = eq if v == "equal" else d[d.method == "allocator"]
            e1 = a[a.w_E == 0.33]
            row = {"allocator inputs": "[no allocator] equal split" if v == "equal" else ABLATION[v][4],
                   "objective": a.obj.mean(), "symbols": a.symbols.mean(), "accuracy (equal w)": e1.acc.mean(),
                   "weighted accuracy": a.wacc.mean(),
                   f"symbols @{lo_snr:g} dB": e1[e1.snr_db == lo_snr].symbols.mean(),
                   f"symbols @{hi_snr:g} dB": e1[e1.snr_db == hi_snr].symbols.mean(),
                   "split spread (tokens std)": e1.groupby(["seed", "snr_db"]).tokens.std().mean(),
                   "ESG tokens (equal → ESG 0.8)": f"{e1.k_E.mean():.2f} → {a[a.w_E == 0.8].k_E.mean():.2f}"}
            if v not in ("full", "equal"):
                ro, ra = self.paired(a, full, "obj"), self.paired(a, full, "wacc")
                row["Δ objective vs full [CI]"] = ci(ro, 4)
                row["Δ weighted acc vs full [CI]"] = ci(ra)
                row["significantly worse"] = "yes" if ro["lo"] > 0 else "no"
            rows.append(row)
        t = pd.DataFrame(rows)
        md(f"Objective = priority-weighted cross-entropy + λ × task tokens (the quantity minimized in "
           f"training, lower is better), on the test set at three priorities (equal, ESG 0.8, Sentiment 0.8).")
        table(t, f"Allocator input ablation (cap {self.s.main_cap}, λ = {lam:g})", 3)
        worse = t[t.get("significantly worse", pd.Series(dtype=str)) == "yes"]["allocator inputs"].tolist()
        self.note("Input ablation: allocators significantly worse than the full one (CLS + SNR + priority): "
                  + ("; ".join(worse) if worse else "none") + ".")
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for v, d in frames.items():
            a = d[(d.method == "allocator") & (d.w_E == 0.33)].groupby("snr_db").symbols.mean()
            ax[0].plot(a.index, a.values, marker="o", ms=3, label=ABLATION[v][4])
        ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("symbols per sentence"); ax[0].set_title("Does it adapt to SNR?")
        ax[0].legend(fontsize=7)
        z = t.sort_values("objective")
        colors = [COLOR["equal"] if s.startswith("[") else COLOR["allocator"] for s in z["allocator inputs"]]
        ax[1].barh(z["allocator inputs"], z["objective"], color=colors)
        ax[1].set_xlim(z.objective.min() * 0.98, z.objective.max() * 1.01); ax[1].invert_yaxis()
        ax[1].set_title("Objective (lower is better)")
        self.save(fig, "6_ablation")

    # ── 7. priority steering ─────────────────────────────────────────────────
    def priority(self):
        lam, out = self.lam, {}
        for mode in ("gain", "embed"):
            out[mode] = self.load(self.s.priority_name(mode, lam), "test", lam)
        rows = []
        for mode, d in out.items():
            a = d[d.method == "allocator"]
            for w, g in a.groupby("w_E"):
                rows.append({"priority form": mode, "w_ESG": w, "ESG tokens": g.k_E.mean(),
                             "ESG accuracy": g.correct_E.mean() * 100, "Sentiment accuracy": g.correct_S.mean() * 100,
                             "FLS accuracy": g.correct_F.mean() * 100, "weighted accuracy": g.wacc.mean()})
        e = out["gain"][out["gain"].method == "equal"]
        for w, g in e.groupby("w_E"):
            rows.append({"priority form": "no allocator (equal split)", "w_ESG": w, "ESG tokens": g.k_E.mean(),
                         "ESG accuracy": g.correct_E.mean() * 100, "Sentiment accuracy": g.correct_S.mean() * 100,
                         "FLS accuracy": g.correct_F.mean() * 100, "weighted accuracy": g.wacc.mean()})
        t = pd.DataFrame(rows)
        md(f"ESG priority swept from {min(self.s.priority_grid)} to {max(self.s.priority_grid)} "
           f"(the other two tasks share the rest), at SNR {', '.join(f'{s:g}' for s in self.s.priority_snrs)} dB, "
           f"cap {self.s.main_cap}. **gain** = score = base + softplus(gain) × w (ours); "
           f"**embed** = priority embedded in the hidden layer (the original figure).")
        table(t, "Priority sweep (test)")
        stats = []
        for mode, d in out.items():
            a = d[d.method == "allocator"].sort_values("w_E")
            rev = a.groupby(["seed", "snr_db", "sentence_id"]).k_E.apply(lambda x: (x.diff() < 0).any()).mean() * 100
            hi, lo = a[a.w_E == a.w_E.max()], a[a.w_E == a.w_E.min()]
            k2 = ["seed", "snr_db", "sentence_id"]
            m = hi[k2 + ["correct_E"]].merge(lo[k2 + ["correct_E"]], on=k2, suffixes=("_hi", "_lo"))
            r = paired_bootstrap(m.correct_E_hi.values * 100, m.correct_E_lo.values * 100)
            stats.append({"priority form": mode, "ESG tokens: lowest → highest w": f"{lo.k_E.mean():.2f} → {hi.k_E.mean():.2f}",
                          "ESG accuracy gain, highest − lowest w [CI]": ci(r),
                          "sentences where more priority gave fewer ESG tokens (%)": rev})
            self.note(f"Priority ({mode}): ESG tokens {lo.k_E.mean():.2f} → {hi.k_E.mean():.2f}, ESG accuracy "
                      f"{ci(r)} points; {rev:.1f}% of sentences received fewer ESG tokens when ESG priority rose.")
        table(pd.DataFrame(stats), "Steering strength and monotonicity")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for mode, c in (("gain", COLOR["allocator"]), ("embed", COLOR["embed"]), ("no allocator (equal split)", COLOR["equal"])):
            z = t[t["priority form"] == mode]
            lab = {"gain": "allocator, gain form (ours)", "embed": "allocator, priority embedding"}.get(mode, mode)
            ax[0].plot(z.w_ESG, z["ESG tokens"], marker="o", color=c, label=lab)
            ax[1].plot(z.w_ESG, z["ESG accuracy"], marker="o", color=c, label=lab)
        ax[0].set_ylabel("ESG task tokens"); ax[1].set_ylabel("ESG accuracy (%)")
        for a in ax:
            a.set_xlabel("ESG priority w_ESG")
        ax[0].legend(fontsize=8); ax[0].set_title("Tokens given to ESG"); ax[1].set_title("ESG accuracy")
        self.save(fig, "7_priority")

    # ── summary ──────────────────────────────────────────────────────────────
    def summary(self):
        md("### Significant results\n" + "\n".join(f"- {f}" for f in self.findings))
        md(f"Figures saved in `{self.fig_dir}`.")
