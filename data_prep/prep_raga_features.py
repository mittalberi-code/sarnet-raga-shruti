"""Compact the 'Indian Art Music Raga Recognition Dataset (features)' zip.

Reads the Zenodo zip directly (no extraction) and writes raga_compact.npz:
tonic-normalised cent contours (downsampled ~23 Hz, int16) + 1-cent pitch
histograms for every recording, plus the dataset's metadata files.
Run:  python prep_raga_features.py
"""
import io, json, os, sys, time, zipfile
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "Indian Art Music Raga Recognition Dataset (features).zip")
OUT = os.path.join(HERE, "raga_compact.npz")
STEP = 10  # keep every 10th pitch frame (~4.4 ms hop -> ~44 ms)

def pick(files, prefs):
    for p in prefs:
        for f in files:
            if f.lower().endswith(p):
                return f
    return None

def main():
    t0 = time.time()
    z = zipfile.ZipFile(ZIP)
    groups = defaultdict(list)
    meta_files = {}
    for n in z.namelist():
        if n.endswith("/") or n.startswith("__MACOSX") or "/._" in n:
            continue
        low = n.lower()
        if low.endswith((".json", ".tsv", ".txt", ".csv", ".md")) and z.getinfo(n).file_size < 3_000_000:
            meta_files[n] = z.read(n).decode("utf-8", "replace")
        stem = n.rsplit(".", 1)[0]  # one group per recording (several recordings can share a folder)
        groups[stem].append(n)
    print("groups:", len(groups), "meta files:", len(meta_files))
    keys, contours, hists, tonics, durs = [], [], [], [], []
    for i, (d, files) in enumerate(sorted(groups.items())):
        pf = pick(files, [".pitchsilintrppp", ".pitch"])
        tf = pick(files, [".tonicfine", ".tonic"])
        if not pf or not tf:
            continue
        try:
            tonic = float(z.read(tf).decode().split()[0])
            raw = z.read(pf).decode()
            first = raw.split("\n", 1)[0].split()
            arr = np.array(raw.split(), dtype=np.float64).reshape(-1, len(first))
            t, f = arr[:, 0], arr[:, 1]
        except Exception as e:
            print("skip", d, e)
            continue
        voiced = f > 20
        cents = np.full(f.shape, -32768, dtype=np.int32)
        cents[voiced] = np.round(1200 * np.log2(f[voiced] / tonic))
        h = np.bincount(np.mod(cents[voiced], 1200), minlength=1200).astype(np.int32)
        c = np.clip(cents[::STEP], -32768, 32767).astype(np.int16)
        keys.append(pf); contours.append(c); hists.append(h); tonics.append(tonic)
        durs.append(float(t[-1]) if len(t) else 0.0)
        if i % 25 == 0:
            print(f"{i}/{len(groups)}  {time.time()-t0:.0f}s  {pf}", flush=True)
    lens = np.array([len(c) for c in contours], dtype=np.int64)
    np.savez_compressed(
        OUT,
        keys=np.array(keys), lens=lens, contours=np.concatenate(contours),
        hists=np.stack(hists), tonics=np.array(tonics), durs=np.array(durs),
        step=STEP, meta=json.dumps(meta_files),
    )
    print("DONE", len(keys), "recordings ->", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB", f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
