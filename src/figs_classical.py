import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import feats as F
from data import load
from classical import svm

OUT = "../figs/"
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888", "axes.labelcolor": "#222",
                     "xtick.color": "#444", "ytick.color": "#444", "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "legend.frameon": False})
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c98500", "#6b6b6b"
R = json.load(open("../data/classical.json"))
TR = [("carnatic", "Carnatic (CMD, 40 rāgas)"), ("hindustani", "Hindustani (HMD, 30 rāgas)")]

# ---- Fig 2: accuracy vs pitch resolution
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
for ax, (t, title) in zip(axs, TR):
    res = R[t]["resolution"]; nb = sorted(int(k) for k in res)
    sv = [100 * res[str(n)]["svm"]["acc"] for n in nb]; sd = [100 * res[str(n)]["svm"]["acc_sd"] for n in nb]
    kn = [100 * res[str(n)]["knn"]["acc"] for n in nb]
    x = np.arange(len(nb))
    ax.errorbar(x, sv, yerr=sd, color=BLUE, lw=2, marker="o", ms=5, capsize=2, label="SVM (RBF)")
    ax.plot(x, kn, color=ORANGE, lw=2, marker="s", ms=5, label="1-NN (Bhattacharyya)")
    td = 100 * R[t]["tdms_knn"]["acc"]
    ax.axhline(td, color=GRAY, lw=1, ls="--"); ax.text(len(nb) - 1, td + 0.3, f"TDMS 1-NN {td:.1f}%", ha="right", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({1200//n} c)" for n in nb], fontsize=7.5)
    ax.set_xlabel("Bins per octave (bin width in cents)"); ax.set_title(title, fontsize=9.5)
    lo = min(min(sv), min(kn)); ax.set_ylim(lo - 3, max(td, max(sv)) + 3)
axs[0].set_ylabel("Accuracy (%)")
axs[0].legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig2_resolution.png", dpi=300); plt.close(fig)

# ---- Fig 3: robustness to simulated pitch correction
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
series = [("pd12_svm", "PD-12 + SVM (clean-trained)", GRAY, "^"),
          ("tdms_knn", "TDMS + 1-NN (clean-trained)", ORANGE, "s"),
          ("pd120_shruti_svm", "PD-120 + shruti + SVM (clean-trained)", YELLOW, "v"),
          ("pd120_shruti_svm_aug", "PD-120 + shruti + SVM (correction-aware)", BLUE, "o")]
for ax, (t, title) in zip(axs, TR):
    rob = R[t]["robustness"]; al = sorted(float(a) for a in rob)
    for k, lab, col, mk in series:
        ax.plot(al, [100 * rob[str(a)][k]["acc"] for a in al], color=col, lw=2, marker=mk, ms=5, label=lab)
    ax.set_xticks(al); ax.set_xlabel("Pitch-correction strength α"); ax.set_title(title, fontsize=9.5)
    ax.set_ylim(0, 100)
axs[0].set_ylabel("Accuracy (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(OUT + "fig3_robustness.png", dpi=300); plt.close(fig)

# ---- data-driven figures
recs, meta, fps = load("../data/raga.npz"); F.FPS = fps
names = json.loads(meta["RagaDataset/Carnatic/_info_/ragaId_to_ragaName_mapping.json"])
rr = [r for r in recs if r["trad"] == "carnatic"]
labs = R["carnatic"]["labels"]; y = np.array([labs.index(r["raga"]) for r in rr]); C = [r["contour"] for r in rr]
# PD-12 confusion -> most confused pair
X12 = np.array([F.pd_feature(c, 12) for c in C]); yp = np.empty_like(y)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X12, y):
    yp[te] = svm().fit(X12[tr], y[tr]).predict(X12[te])
cm12 = confusion_matrix(y, yp); M = cm12 + cm12.T; np.fill_diagonal(M, 0)
i, j = np.unravel_index(np.argmax(M), M.shape)
pair = (labs[i], labs[j]); pair_names = (names.get(pair[0], pair[0]), names.get(pair[1], pair[1]))
json.dump({"pair": pair_names, "confusions_pd12": int(M[i, j])}, open("../data/pair.json", "w"), ensure_ascii=False)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
for idx, col, nm in [(i, BLUE, pair_names[0]), (j, ORANGE, pair_names[1])]:
    H = np.mean([np.bincount(np.mod(F.voiced(C[k]).astype(int), 1200), minlength=1200) / max(len(F.voiced(C[k])), 1)
                 for k in np.where(y == idx)[0]], axis=0)
    H = gaussian_filter1d(H, 4, mode="wrap") * 100
    ax.plot(np.arange(1200), H, color=col, lw=1.6, label=nm)
for s in range(12):
    ax.axvline(s * 100, color="#bbb", lw=0.6, ls=":")
ax.set_xticks(np.arange(0, 1200, 100)); ax.set_xticklabels(["S", "r", "R", "g", "G", "M", "m", "P", "d", "D", "n", "N"])
ax.set_xlim(0, 1199); ax.set_xlabel("Pitch relative to tonic (cents, octave-folded); dotted lines = equal-tempered svaras")
ax.set_ylabel("Share of voiced frames (%)"); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "fig5_pair.png", dpi=300); plt.close(fig)

# ---- Fig 6: contour excerpt before/after correction
k = np.where(y == i)[0][0]; c = C[k]
# pick a 12-s window with most voiced frames
L = int(4 * fps)
def score(s):
    w = c[s:s + L]
    if (w == F.UNV).any(): return -1
    return np.ptp(w) < 700 and np.abs(np.diff(w.astype(float))).mean()
best = max(range(0, len(c) - L, L // 4), key=score)
t = np.arange(L) / fps
fig, ax = plt.subplots(figsize=(7.2, 2.6))
for a, col, lab in [(0.0, BLUE, "Original (α = 0)"), (0.5, YELLOW, "α = 0.5"), (1.0, ORANGE, "Fully corrected (α = 1)")]:
    cc = F.pitch_correct(c, a)[best:best + L].astype(float); cc[cc == F.UNV] = np.nan
    ax.plot(t, cc, color=col, lw=1.4 if a else 2.2, label=lab, drawstyle="default")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Pitch (cents re tonic)"); ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
lo, hi = ax.get_ylim()
for v in range(int(np.floor(lo / 100) * 100), int(hi) + 1, 100):
    ax.axhline(v, color="#ccc", lw=0.5, ls=":")
fig.tight_layout(); fig.savefig(OUT + "fig6_contour.png", dpi=300); plt.close(fig)
print("pair", pair_names, M[i, j])
