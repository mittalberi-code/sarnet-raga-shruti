import json, re
import numpy as np


def load(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    lens = z["lens"]; flat = z["contours"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    contours = [flat[offs[i]:offs[i + 1]].astype(np.int32) for i in range(len(keys))]
    meta = json.loads(str(z["meta"]))
    durs = z["durs"]
    recs = []
    for k, c, d in zip(keys, contours, durs):
        parts = k.replace("\\", "/").split("/")
        low = k.lower()
        trad = "carnatic" if "carnatic" in low else ("hindustani" if "hindustani" in low else "unknown")
        # layout: .../features/<raga>/<recording>/<file>
        if "features" in parts:
            i = parts.index("features"); raga = parts[i + 1]; artist = parts[i + 2]; rec = parts[-2]
        else:
            raga = parts[-3]; rec = parts[-2]; artist = rec
        recs.append(dict(key=k, trad=trad, raga=raga, artist=artist, rec=rec, contour=c, dur=float(d)))
    fps = float(np.median([len(r["contour"]) / r["dur"] for r in recs if r["dur"] > 0]))
    return recs, meta, fps
