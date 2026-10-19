"""Timing measurements + confusion-matrix figure for the best CMD system (run after deep.py)."""
import json, time, re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import feats as F, deep as D
from data import load
from classical import svm

recs, meta, fps = load("../data/raga.npz")
rr = [r for r in recs if r["trad"] == "carnatic"]
labs = sorted(set(r["raga"] for r in rr)); y = np.array([labs.index(r["raga"]) for r in rr])
names = json.loads(meta["RagaDataset/Carnatic/_info_/ragaId_to_ragaName_mapping.json"])
E = {}
# --- timing (full-rate contours)
F.FPS = fps
C = [r["contour"] for r in rr]
idx = np.random.default_rng(0).choice(len(C), 40, replace=False)
t = time.time(); [np.concatenate([F.pd_feature(C[i], 120), F.shruti_descriptors(C[i])]) for i in idx]; E["t_feat_ms"] = 1000 * (time.time() - t) / len(idx)
t = time.time(); [F.tdms(C[i]) for i in idx]; E["t_tdms_ms"] = 1000 * (time.time() - t) / len(idx)
X = np.array([np.concatenate([F.pd_feature(c, 120), F.shruti_descriptors(c)]) for c in C])
tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
t = time.time(); svm().fit(X[tr], y[tr]); E["t_svm_fit_s"] = time.time() - t
# SAR-Net fold time from log
log = open("../data/deep.log").read()
m = re.findall(r"carnatic sarnet_full fold (\d) (\d+)s", log)
if m: E["sarnet_fold_min"] = int(m[-1][1]) / len(m) / 60
# SAR-Net inference time: train a tiny 1-epoch model just to time inference
F.FPS = fps / D.DS
Cd = [c[::D.DS] for c in C]
REC = np.array([np.concatenate([F.pd_feature(c, 120), F.shruti_descriptors(c)]) for c in Cd])
model = D.SARNet(len(labs), REC.shape[1]); model.eval(); L = int(D.EXC_S * F.FPS); rng = np.random.default_rng(0)
t = time.time()
with torch.no_grad():
    for i in idx[:20]:
        e = torch.tensor(D.excerpts(Cd[i], L, 4, rng, all_=True)); model(e, [len(e)], torch.zeros(1, REC.shape[1]))
E["t_sarnet_inf_ms"] = 1000 * (time.time() - t) / 20
E["n_params"] = int(sum(p.numel() for p in model.parameters()))

# --- confusion matrix of SAR-Net (full) on CMD
DP = json.load(open("../data/deep.json"))
P = np.array(DP["carnatic"]["sarnet_full"]["clean"]["prob"]); yd = np.array(DP["carnatic"]["y"])
cm = confusion_matrix(yd, P.argmax(1), labels=range(len(labs)))
order = np.argsort([names.get(l, l) for l in labs])
cmo = cm[np.ix_(order, order)]; lab_o = [names.get(labs[k], labs[k]) for k in order]
fig, ax = plt.subplots(figsize=(7.4, 7.0))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
im = ax.imshow(cmo, cmap=cmap, vmin=0, vmax=12)
ax.set_xticks(range(len(lab_o))); ax.set_yticks(range(len(lab_o)))
ax.set_xticklabels(lab_o, rotation=90, fontsize=6.2); ax.set_yticklabels(lab_o, fontsize=6.2)
for a in range(len(lab_o)):
    for b in range(len(lab_o)):
        if cmo[a, b] and a != b: ax.text(b, a, cmo[a, b], ha="center", va="center", fontsize=5.5, color="#111")
ax.set_xlabel("Predicted rāga"); ax.set_ylabel("True rāga")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Recordings (12 per rāga)", fontsize=7); cb.ax.tick_params(labelsize=6)
ax.grid(False)
fig.tight_layout(); fig.savefig("../figs/fig7_confusion.png", dpi=300); plt.close(fig)
off = cm.copy(); np.fill_diagonal(off, 0); S = off + off.T
pairs = []
for _ in range(3):
    a, b = np.unravel_index(np.argmax(S), S.shape); pairs.append((names.get(labs[a], labs[a]), names.get(labs[b], labs[b]), int(S[a, b]))); S[a, b] = S[b, a] = 0
perfect = int(sum(cm[k, k] == 12 for k in range(len(labs))))
E["confusion_text"] = (f"Figure 6 shows the confusion matrix of SAR-Net on the Carnatic dataset. {perfect} of the 40 rāgas were recognised "
                       f"without error. The most frequent confusions were between {pairs[0][0]} and {pairs[0][1]} ({pairs[0][2]} recordings), "
                       f"{pairs[1][0]} and {pairs[1][1]} ({pairs[1][2]}), and {pairs[2][0]} and {pairs[2][1]} ({pairs[2][2]}); "
                       f"such pairs share most of their svaras and differ mainly in characteristic phrases and intonation.")
E["confusion_caption"] = "Confusion matrix of SAR-Net on the Carnatic dataset (40 rāgas, alphabetical; off-diagonal counts shown)"
json.dump(E, open("../data/extra.json", "w"), ensure_ascii=False, indent=1)
print(E)

# --- late fusion: SAR-Net probabilities + TDMS class similarities (Borda rank fusion, no tuned weights)
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, f1_score
F.FPS = fps
for trad in ["carnatic", "hindustani"]:
    r2 = [r for r in recs if r["trad"] == trad]
    lb = sorted(set(r["raga"] for r in r2)); yy = np.array([lb.index(r["raga"]) for r in r2])
    T = np.sqrt(np.array([F.tdms(r["contour"]) for r in r2]))
    Pd = np.array(DP[trad]["sarnet_full"]["clean"]["prob"])
    assert (np.array(DP[trad]["y"]) == yy).all()
    S = np.zeros_like(Pd)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(T, yy):
        sim = T[te] @ T[tr].T
        for k in range(len(lb)):
            S[te, k] = sim[:, yy[tr] == k].max(1)
    fused = rankdata(Pd, axis=1) + rankdata(S, axis=1)
    yp = fused.argmax(1)
    E[f"fusion_{trad}"] = {"acc": float(accuracy_score(yy, yp)), "f1": float(f1_score(yy, yp, average="macro"))}
    # McNemar: fusion vs TDMS alone (seed-0 folds)
    yt = S.argmax(1); b = int(((yp == yy) & (yt != yy)).sum()); c = int(((yp != yy) & (yt == yy)).sum())
    from scipy.stats import binomtest
    E[f"mcnemar_{trad}"] = {"b": b, "c": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
                            "tdms_seed0_acc": float(accuracy_score(yy, yt))}
json.dump(E, open("../data/extra.json", "w"), ensure_ascii=False, indent=1)
print({k: v for k, v in E.items() if k.startswith(("fusion", "mcnemar"))})
