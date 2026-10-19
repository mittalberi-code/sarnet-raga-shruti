"""SAR-Net: shruti-aware rāga network (CPU). Contour branch (1-D CNN + BiGRU + attention
over 30-s excerpts, mean-pooled per recording) fused with a recording-level branch on the
120-bin pitch distribution and 84-d shruti descriptors."""
import json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import feats as F
from data import load

torch.set_num_threads(2)
DS = 2           # extra temporal decimation of compact contour
EXC_S = 30.0     # excerpt length (s)
NB = 120         # pitch bins (10 cents)


class ContourBranch(nn.Module):
    def __init__(self, nb=NB, h=64):
        super().__init__()
        self.c1 = nn.Conv1d(nb, 96, 5, padding=2); self.b1 = nn.BatchNorm1d(96)
        self.c2 = nn.Conv1d(96, 128, 5, padding=2); self.b2 = nn.BatchNorm1d(128)
        self.gru = nn.GRU(128, h, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * h, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):  # x: (B, nb, L)
        x = Fn.max_pool1d(Fn.relu(self.b1(self.c1(x))), 2)
        x = Fn.max_pool1d(Fn.relu(self.b2(self.c2(x))), 2)
        x, _ = self.gru(self.drop(x.transpose(1, 2)))
        a = torch.softmax(self.att(x).squeeze(-1), 1)
        return (x * a.unsqueeze(-1)).sum(1)  # (B, 2h)


class SARNet(nn.Module):
    def __init__(self, n_cls, d_rec, use_contour=True, use_rec=True):
        super().__init__()
        self.uc, self.ur = use_contour, use_rec
        dim = 0
        if use_contour:
            self.cb = ContourBranch(); dim += 128
        if use_rec:
            self.rb = nn.Sequential(nn.Linear(d_rec, 128), nn.ReLU(), nn.Dropout(0.3)); dim += 128
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n_cls))

    def forward(self, ex, nex, rec):  # ex: (sum K, nb, L); nex: list of K per recording
        z = []
        if self.uc:
            e = self.cb(ex)
            z.append(torch.stack([t.mean(0) for t in torch.split(e, nex)]))
        if self.ur:
            z.append(self.rb(rec))
        return self.head(torch.cat(z, 1))


def excerpts(c, L, K, rng, all_=False):
    n = len(c)
    if n <= L:
        starts = [0]
    elif all_:
        starts = list(range(0, n - L + 1, L))[:24]
    else:
        starts = rng.integers(0, n - L + 1, size=K)
    return np.stack([F.excerpt_roll(c, s, L, NB) for s in starts])


def train_eval(C, REC, y, tr, te, n_cls, use_contour=True, use_rec=True, epochs=60, seed=0,
               test_C=None, test_REC=None, aug=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    L = int(EXC_S * F.FPS)
    sc = StandardScaler().fit(REC[tr])
    Rtr = torch.tensor(sc.transform(REC[tr]), dtype=torch.float32)
    if aug is not None:  # correction-aware training: each sample drawn at a random correction strength
        AK = sorted(aug); RtrA = {a: torch.tensor(sc.transform(aug[a][1][tr]), dtype=torch.float32) for a in AK}
    model = SARNet(n_cls, REC.shape[1], use_contour, use_rec)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * int(np.ceil(len(tr) / 16)))
    ytr = torch.tensor(y[tr])
    K = 4
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(tr))
        for b in range(0, len(tr), 16):
            idx = perm[b:b + 16]
            if aug is not None:
                al = [AK[k] for k in rng.integers(0, len(AK), size=len(idx))]
                srcC = [aug[a][0][tr[i]] for a, i in zip(al, idx)]
                rec_b = torch.stack([RtrA[a][i] for a, i in zip(al, idx)])
            else:
                srcC = [C[tr[i]] for i in idx]; rec_b = Rtr[idx]
            if use_contour:
                exs = [excerpts(c, L, K, rng) for c in srcC]
                ex = torch.tensor(np.concatenate(exs)); nex = [len(e) for e in exs]
            else:
                ex, nex = None, None
            out = model(ex, nex, rec_b)
            loss = Fn.cross_entropy(out, ytr[idx], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    def predict(Cs, RECs):
        P = []
        Rte = torch.tensor(sc.transform(RECs[te]), dtype=torch.float32)
        with torch.no_grad():
            for j, i in enumerate(te):
                if use_contour:
                    e = torch.tensor(excerpts(Cs[i], L, K, rng, all_=True)); nex = [len(e)]
                else:
                    e, nex = None, None
                P.append(torch.softmax(model(e, nex, Rte[j:j + 1]), 1).numpy()[0])
        return np.array(P)
    out = {"clean": predict(C, REC)}
    if test_C is not None:
        for a, (Ca, RECa) in test_C.items():
            out[a] = predict(Ca, RECa)
    return out


def run(npz, out_json, which="both", epochs=60):
    recs, meta, fps = load(npz)
    F.FPS = fps / DS
    R = {}
    for trad in ["carnatic", "hindustani"]:
        if which not in ("both", trad):
            continue
        rr = [r for r in recs if r["trad"] == trad]
        labs = sorted(set(r["raga"] for r in rr)); y = np.array([labs.index(r["raga"]) for r in rr])
        C = [r["contour"][::DS] for r in rr]
        REC = np.array([np.concatenate([F.pd_feature(c, 120), F.shruti_descriptors(c)]) for c in C])
        alphas = [0.25, 0.5, 0.75, 1.0]
        CA = {a: [F.pitch_correct(c, a) for c in C] for a in alphas}
        RA = {a: np.array([np.concatenate([F.pd_feature(c, 120), F.shruti_descriptors(c)]) for c in CA[a]]) for a in alphas}
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        res = {}
        AUG = {0.0: (C, REC), **{a: (CA[a], RA[a]) for a in alphas}}
        for name, uc, ur, ag in [("sarnet_full", True, True, False), ("contour_only", True, False, False),
                                 ("sarnet_aug", True, True, True)]:
            P = {k: np.zeros((len(y), len(labs))) for k in ["clean"] + alphas}
            t0 = time.time()
            for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
                o = train_eval(C, REC, y, tr, te, len(labs), uc, ur, epochs=epochs,
                               test_C={a: (CA[a], RA[a]) for a in alphas} if name != "contour_only" else None,
                               aug=AUG if ag else None)
                for k, v in o.items():
                    P[k][te] = v
                print(trad, name, "fold", f, f"{time.time()-t0:.0f}s",
                      accuracy_score(y[te], o["clean"].argmax(1)), flush=True)
            res[name] = {str(k): {"acc": float(accuracy_score(y, P[k].argmax(1))),
                                  "f1": float(f1_score(y, P[k].argmax(1), average="macro")),
                                  "prob": P[k].tolist()} for k in P if name != "contour_only" or k == "clean"}
            print(trad, name, {k: v["acc"] for k, v in res[name].items()}, flush=True)
        res["y"] = y.tolist(); res["labels"] = labs
        R[trad] = res
        json.dump(R, open(out_json, "w"))
    print("DONE")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "both",
        int(sys.argv[4]) if len(sys.argv) > 4 else 60)
