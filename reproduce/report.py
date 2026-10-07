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
    def __init__(self, study, prefix=""):
        self.s = study
        self.prefix = prefix                      # figure-name prefix (e.g. "A_" for the small budget)
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
        path = os.path.join(self.fig_dir, f"{self.prefix}{name}.png")
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


# ════════════════════════════════════════════════════════════════════════════
# Part A: small token budget (4 task tokens per task), see runner.SmallStudy
# ════════════════════════════════════════════════════════════════════════════

def symbols_for(points, target):
    """Symbols needed to reach `target` accuracy along the upper envelope of (symbols, acc) points."""
    env, best = [], -np.inf
    for s, a in sorted(points):
        if a > best:
            env.append((s, a))
            best = a
    if not env or target > env[-1][1]:
        return np.nan
    if target <= env[0][1]:
        return env[0][0]
    for (s0, a0), (s1, a1) in zip(env, env[1:]):
        if a0 < target <= a1:
            return s0 + (target - a0) / (a1 - a0) * (s1 - s0)
    return np.nan


class SmallReport(Report):
    def __init__(self, study):
        super().__init__(study, prefix="A_")
        self.lam = 0.0

    def save(self, fig, name):
        return super().save(fig, {"7_priority": "3_priority", "3_rate": "2_rate"}.get(name, name))

    def _done(self, name):
        return bool(glob.glob(os.path.join(self.s.run_dir(name), "seed*", "per_sentence.csv.gz")))

    def _task_curves(self):
        """Per-task accuracy (%) for k = 1..J tokens, per SNR, from the stage-1 evaluation."""
        p = self.load(self.s.PIPELINE)
        u = p[p.method.str.startswith("uniform_k")].copy()
        u["k"] = u.method.str[9:].astype(int)
        return u, {key: u.pivot_table(index="k", columns="snr_db", values=f"correct_{key}") * 100 for key in "SFE"}

    # ── A1. saturation: gain from 1 to J tokens per task ─────────────────────
    def saturation(self):
        super().saturation()
        u, cur = self._task_curves()
        J = int(u.k.max())
        rows = []
        for s in cur["S"].columns:
            rows.append({"SNR (dB)": s, **{name: cur[key].loc[J, s] - cur[key].loc[1, s]
                                         for key, name in (("S", "Sentiment"), ("F", "FLS"), ("E", "ESG"))}})
        table(pd.DataFrame(rows), f"Accuracy gain (points) from 1 to {J} task tokens per task, no allocator", 1)

    # ── A2. fixed cap 8, lambda = 0 ──────────────────────────────────────────
    def fixed_cap(self):
        from reproduce.runner import CAP8_VARIANTS
        cap = self.s.fixed_cap
        runs = {v: self.load(self.s.cap8_name(v)) for v in CAP8_VARIANTS if self._done(self.s.cap8_name(v))}
        base = runs["gain"]
        eq = base[base.method == "equal"]
        groups = [("Equal split (no allocator)", eq),
                  ("Proportional split", base[base.method == "proportional"]),
                  ("Random split, same total as the allocator", base[base.method == "random_matched"])]
        groups += [(CAP8_VARIANTS[v][1], d[d.method == "allocator"]) for v, d in runs.items()]
        groups += [("Hindsight oracle (not practical)", base[base.method == "oracle_greedy"])]
        rows = []
        for name, x in groups:
            row = {"method": name, "symbols": x.symbols.mean(), "Sentiment": x.correct_S.mean() * 100,
                   "FLS": x.correct_F.mean() * 100, "ESG": x.correct_E.mean() * 100, "mean accuracy": x.acc.mean()}
            if not name.startswith("Equal"):
                r = self.paired(x, eq, "acc")
                per_seed = (x.groupby("seed").acc.mean() - eq.groupby("seed").acc.mean()).round(2).tolist()
                row["Δ vs equal [95% CI]"] = ci(r)
                row["Δ per seed"] = ", ".join(f"{v:+.2f}" for v in per_seed)
            rows.append(row)
        table(pd.DataFrame(rows), f"Fixed cap {cap}, λ = 0 (test, mean of seeds and SNRs)")
        for name, x in groups[3:-1]:
            r = self.paired(x, eq, "acc")
            self.note(f"Cap {cap}: {name} vs equal split: {ci(r)} points of accuracy.")
        # best fixed split chosen in hindsight, per SNR (each task decodes only its own tokens)
        u, cur = self._task_curves()
        J = int(u.k.max())
        combos = [(a, b, c) for a in range(1, J + 1) for b in range(1, J + 1) for c in range(1, J + 1)
                  if a + b + c == cap]
        rows = []
        for s in cur["S"].columns:
            score = {k: (cur["S"].loc[k[0], s] + cur["F"].loc[k[1], s] + cur["E"].loc[k[2], s]) / 3 for k in combos}
            best = max(score, key=score.get)
            e = eq[eq.snr_db == s].acc.mean()
            rows.append({"SNR (dB)": s, "best split S/F/E": "/".join(map(str, best)),
                         "best split accuracy": score[best], "equal split accuracy": e,
                         "room (points)": score[best] - e})
        t = pd.DataFrame(rows)
        table(t, f"How much room is there? Best fixed split per SNR, chosen with the test labels (cap {cap})")
        self.note(f"Cap {cap}: the best fixed split chosen in hindsight beats the equal split by at most "
                  f"{t['room (points)'].max():.2f} points (at {t.loc[t['room (points)'].idxmax(), 'SNR (dB)']:g} dB).")

    # ── A3. variable rate at cap 12 ──────────────────────────────────────────
    def rate(self):
        u, _ = self._task_curves()
        fixed = [(f"Fixed {k}/{k}/{k} (no allocator)", 12 * k, g.acc.mean()) for k, g in u.groupby("k")]
        first = self.load(self.s.rate_name(self.s.lambda_grid[0]))
        for m in ("fixed:4-3-3", "fixed:3-3-2"):
            x = first[first.method == m]
            fixed.append((f"Fixed {m[6:].replace('-', '/')} (no allocator)", x.symbols.mean(), x.acc.mean()))
        rows = [{"method": n, "λ": "", "symbols": s, "accuracy": a}
                for n, s, a in sorted(fixed, key=lambda z: -z[1])]
        curves = {}
        for conf in (False, True):
            label = "Learned allocator + confidence" if conf else "Learned allocator"
            pts = []
            for lam in self.s.lambda_grid:
                name = self.s.rate_name(lam, conf)
                if not self._done(name):
                    continue
                a = self.load(name, lam=lam)
                a = a[a.method == "allocator"]
                pts.append((a.symbols.mean(), a.acc.mean()))
                rows.append({"method": label, "λ": f"{lam:g}", "symbols": pts[-1][0], "accuracy": pts[-1][1]})
            if pts:
                curves[label] = pts
        table(pd.DataFrame(rows), f"Accuracy vs symbols, cap {self.s.rate_cap} (test)", 2)
        fpts = [(s, a) for _, s, a in fixed]
        lo = max(min(a for _, a in p) for p in [fpts, *curves.values()])
        hi = min(max(a for _, a in p) for p in [fpts, *curves.values()])
        targets = np.round(np.linspace(lo + 0.1 * (hi - lo), hi - 0.02 * (hi - lo), 5), 1)
        trows = []
        for tg in targets:
            r = {"target accuracy (%)": tg, "fixed splits": symbols_for(fpts, tg)}
            for label, p in curves.items():
                r[label] = symbols_for(p, tg)
            best = np.nanmin([r[label] for label in curves])
            r["saving (%)"] = (1 - best / r["fixed splits"]) * 100
            trows.append(r)
        t = pd.DataFrame(trows)
        table(t, "Symbols per sentence needed to reach a target accuracy (along each curve)", 1)
        lo_s, hi_s = t["saving (%)"].min(), t["saving (%)"].max()
        self.note(f"Variable rate (cap {self.s.rate_cap}): to reach the same accuracy as fixed splits, the "
                  f"allocator changes the symbols needed by {-hi_s:+.0f} to {-lo_s:+.0f}% (negative = fewer).")
        lam = 0.01 if 0.01 in self.s.lambda_grid else self.s.lambda_grid[len(self.s.lambda_grid) // 2]
        d = self.load(self.s.rate_name(lam), lam=lam)
        f, a = d[d.method == "fixed:3-3-2"], d[d.method == "allocator"]
        per = pd.DataFrame({"fixed 3/3/2: symbols": f.groupby("snr_db").symbols.mean(),
                            "fixed 3/3/2: accuracy": f.groupby("snr_db").acc.mean(),
                            "allocator: symbols": a.groupby("snr_db").symbols.mean(),
                            "allocator: accuracy": a.groupby("snr_db").acc.mean()}).reset_index()
        table(per.rename(columns={"snr_db": "SNR (dB)"}), f"Per SNR at λ = {lam:g} (test)", 1)
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        fs = sorted(fpts)
        ax.scatter([s for s, _ in fs], [a for _, a in fs], color=COLOR["equal"], zorder=3,
                   label="fixed splits (no allocator)")
        for (label, p), c in zip(curves.items(), (COLOR["allocator"], COLOR["embed"])):
            p = sorted(p)
            ax.plot([s for s, _ in p], [a for _, a in p], marker="o", color=c, label=label)
        ax.set_xlabel("average symbols per sentence"); ax.set_ylabel("mean accuracy (%)")
        ax.set_title(f"Accuracy vs bandwidth, cap {self.s.rate_cap} (test)"); ax.legend(fontsize=8)
        self.save(fig, "3_rate")

    # ── A4. priority (gain vs embedding) and the fixed-priority check ────────
    def priority(self):
        super().priority()
        if not self._done(self.s.FIXED_PRIORITY):
            return
        fp = self.load(self.s.FIXED_PRIORITY)
        fp = fp[fp.method == "allocator"]
        w_e = fp.w_E.iloc[0]
        g = self.load(self.s.priority_name("gain"))
        g = g[(g.method == "allocator") & (g.w_E == w_e) & (g.snr_db.isin(fp.snr_db.unique()))]
        rows = [{"allocator": f"trained only at w = ({fp.w_S.iloc[0]:g}, {fp.w_F.iloc[0]:g}, {w_e:g})",
                 "ESG tokens": fp.k_E.mean(), "ESG accuracy": fp.correct_E.mean() * 100,
                 "weighted accuracy": fp.wacc.mean()},
                {"allocator": "trained with random priorities (one model for all)",
                 "ESG tokens": g.k_E.mean(), "ESG accuracy": g.correct_E.mean() * 100,
                 "weighted accuracy": g.wacc.mean()}]
        r = self.paired(g, fp, "wacc")
        snrs = ", ".join(f"{s:g}" for s in fp.snr_db.unique())
        table(pd.DataFrame(rows), f"One model for every priority? (test, {snrs} dB)")
        md(f"Weighted accuracy, random-priority model minus fixed-priority model: {ci(r)} points.")
        self.note(f"A single allocator trained with random priorities vs one trained only at ESG priority "
                  f"{w_e:g}: weighted accuracy {ci(r)} points.")

    # ── A5. joint training (optional) ────────────────────────────────────────
    def joint(self):
        if not self._done(self.s.JOINT):
            md("Joint-training run not found (it is optional).")
            return
        d = self.load(self.s.JOINT)
        a, e = d[d.method == "allocator"], d[d.method == "equal"]
        r = self.paired(a, e, "acc")
        table(pd.DataFrame([{"method": "allocator trained jointly with the pipeline", "task tokens": a.tokens.mean(),
                             "ESG tokens": a.k_E.mean(), "mean accuracy": a.acc.mean()},
                            {"method": "equal split (same pipeline)", "task tokens": e.tokens.mean(),
                             "ESG tokens": e.k_E.mean(), "mean accuracy": e.acc.mean()}]),
              f"Single-stage (joint) training, cap {self.s.fixed_cap}")
        md(f"Allocator minus equal split: {ci(r)} points. If the allocator's split stays at the equal split, "
           f"the joint schedule did not let it learn (the training log shows the selected epoch).")
