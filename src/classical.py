"""Non-deep experiments: pitch-distribution resolution ablation, TDMS baseline,
shruti descriptors, excerpt-length ablation, pitch-correction robustness."""
import json, sys, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
import feats as F
from data import load

SEEDS = [0, 1, 2]
NFOLD = 5
TRADS = ["carnatic", "hindustani"]


def knn1(Xtr, ytr, Xte, metric="bhatt"):
    # Bhattacharyya similarity on sqrt-normalised vectors == dot product
    S = Xte @ Xtr.T
    return ytr[np.argmax(S, axis=1)], S


def svm():
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", kernel="rbf"))


def cv_group(X, y, groups, kind):
    sgk = StratifiedGroupKFold(NFOLD, shuffle=True, random_state=0)
    yp = np.empty_like(y)
    for tr, te in sgk.split(X, y, groups):
        if kind == "knn":
            yp[te], _ = knn1(X[tr], y[tr], X[te])
        else:
            yp[te] = svm().fit(X[tr], y[tr]).predict(X[te])
    return dict(acc=float(accuracy_score(y, yp)), f1=float(f1_score(y, yp, average="macro")))


def cv_eval(X, y, kind, seeds=SEEDS, test_transform=None, return_pred=False):
    accs, f1s = [], []
    preds = {}
    for s in seeds:
        skf = StratifiedKFold(NFOLD, shuffle=True, random_state=s)
        yp = np.empty_like(y)
        for tr, te in skf.split(X if not callable(X) else np.zeros(len(y)), y):
            Xtr = X[tr]
            Xte = X[te] if test_transform is None else test_transform(te)
            if kind == "knn":
                yp[te], _ = knn1(Xtr, y[tr], Xte)
            else:
                m = svm().fit(Xtr, y[tr]); yp[te] = m.predict(Xte)
        accs.append(accuracy_score(y, yp)); f1s.append(f1_score(y, yp, average="macro"))
        preds[s] = yp
    out = dict(acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), f1=float(np.mean(f1s)), f1_sd=float(np.std(f1s)))
    if return_pred:
        out["pred"] = preds
    return out


def run(npz, out_json):
    recs, meta, fps = load(npz)
    F.FPS = fps
    print("fps", fps, "recs", len(recs), flush=True)
    R = {"fps": fps}
    for trad in TRADS:
        rr = [r for r in recs if r["trad"] == trad]
        if not rr:
            continue
        labs = sorted(set(r["raga"] for r in rr)); y = np.array([labs.index(r["raga"]) for r in rr])
        C = [r["contour"] for r in rr]
        res = {"n_rec": len(rr), "n_raga": len(labs), "hours": float(sum(r["dur"] for r in rr) / 3600)}
        t0 = time.time()
        # --- A. resolution ablation
        resA = {}
        for nb in [12, 24, 36, 72, 120, 240, 1200]:
            X = np.array([F.pd_feature(c, nb, smooth_cents=5 if nb >= 240 else 0) for c in C])
            resA[nb] = {"knn": cv_eval(X, y, "knn"), "svm": cv_eval(X, y, "svm")}
            print(trad, "PD", nb, resA[nb]["knn"]["acc"], resA[nb]["svm"]["acc"], f"{time.time()-t0:.0f}s", flush=True)
        res["resolution"] = resA
        # --- B. TDMS baseline
        Xt = np.sqrt(np.array(Parallel(n_jobs=2)(delayed(F.tdms)(c) for c in C)))
        res["tdms_knn"] = cv_eval(Xt, y, "knn", return_pred=True)
        print(trad, "TDMS", res["tdms_knn"]["acc"], flush=True)
        # --- C. shruti descriptors
        Xs = np.array(Parallel(n_jobs=2)(delayed(F.shruti_descriptors)(c) for c in C))
        X120 = np.array([F.pd_feature(c, 120) for c in C])
        res["shruti_svm"] = cv_eval(Xs, y, "svm")
        res["pd120_shruti_svm"] = cv_eval(np.hstack([X120, Xs]), y, "svm", return_pred=True)
        print(trad, "shruti", res["shruti_svm"]["acc"], "pd120+shruti", res["pd120_shruti_svm"]["acc"], flush=True)
        # --- C2. artist-disjoint folds (no performer shared between train and test)
        groups = np.array([r["artist"] for r in rr])
        X12 = np.array([F.pd_feature(c, 12) for c in C])
        res["artist_disjoint"] = {"pd12_svm": cv_group(X12, y, groups, "svm"),
                                  "tdms_knn": cv_group(Xt, y, groups, "knn"),
                                  "pd120_shruti_svm": cv_group(np.hstack([X120, Xs]), y, groups, "svm")}
        res["n_artists"] = int(len(set(groups)))
        print(trad, "artist-disjoint", res["artist_disjoint"], flush=True)
        # --- D. excerpt length (first L seconds)
        resD = {}
        for L in [30, 60, 120, 300]:
            n = int(L * fps)
            Xl = np.array([F.pd_feature(c[:n], 120) for c in C]); Xsl = np.array([F.shruti_descriptors(c[:n]) for c in C])
            Xtl = np.sqrt(np.array([F.tdms(c[:n]) for c in C]))
            resD[L] = {"pd120_shruti_svm": cv_eval(np.hstack([Xl, Xsl]), y, "svm", seeds=[0]),
                       "tdms_knn": cv_eval(Xtl, y, "knn", seeds=[0])}
            print(trad, "len", L, resD[L], flush=True)
        res["length"] = resD
        # --- E. pitch-correction robustness
        ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
        FE = {}
        for a in ALPHAS:
            Cc = [F.pitch_correct(c, a) for c in C]
            FE[a] = dict(mdi=float(np.nanmean([F.microtonal_deviation(c) for c in Cc])),
                         P12=np.array([F.pd_feature(c, 12) for c in Cc]),
                         PS=np.hstack([np.array([F.pd_feature(c, 120) for c in Cc]),
                                       np.array(Parallel(n_jobs=2)(delayed(F.shruti_descriptors)(c) for c in Cc))]),
                         T=np.sqrt(np.array(Parallel(n_jobs=2)(delayed(F.tdms)(c) for c in Cc))))
        def aug_eval(a):
            skf = StratifiedKFold(NFOLD, shuffle=True, random_state=0); yp = np.empty_like(y)
            for tr, te in skf.split(np.zeros(len(y)), y):
                Xtr = np.vstack([FE[b]["PS"][tr] for b in ALPHAS]); ytr = np.concatenate([y[tr]] * len(ALPHAS))
                yp[te] = svm().fit(Xtr, ytr).predict(FE[a]["PS"][te])
            return dict(acc=float(accuracy_score(y, yp)), f1=float(f1_score(y, yp, average="macro")))
        resE = {}
        for a in ALPHAS:
            resE[a] = {
                "mdi_cents": FE[a]["mdi"],
                "pd12_svm": cv_eval(FE[0.0]["P12"], y, "svm", seeds=[0], test_transform=lambda te: FE[a]["P12"][te]),
                "tdms_knn": cv_eval(FE[0.0]["T"], y, "knn", seeds=[0], test_transform=lambda te: FE[a]["T"][te]),
                "pd120_shruti_svm": cv_eval(FE[0.0]["PS"], y, "svm", seeds=[0], test_transform=lambda te: FE[a]["PS"][te]),
                "pd120_shruti_svm_matched": cv_eval(FE[a]["PS"], y, "svm", seeds=[0]),
                "tdms_knn_matched": cv_eval(FE[a]["T"], y, "knn", seeds=[0]),
                "pd120_shruti_svm_aug": aug_eval(a),
            }
            print(trad, "alpha", a, {k: (v if isinstance(v, float) else v["acc"]) for k, v in resE[a].items()}, flush=True)
        res["robustness"] = resE
        res["labels"] = labs
        R[trad] = res
        json.dump(R, open(out_json, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("DONE")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        TRADS = sys.argv[3].split(",")
    run(sys.argv[1], sys.argv[2])
