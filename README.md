# Deep Learning-Driven Rāga Identification and Microtonal Shruti Pitch-Contour Modeling

Code and results for the manuscript *Deep Learning-Driven Rāga Identification and Microtonal Shruti
Pitch-Contour Modeling in Indian Classical Music Production* (H. K. Mittal, submitted to
*ShodhKosh: Journal of Visual and Performing Arts*).

The code reproduces every table and figure in the paper on an ordinary CPU (no GPU needed):

- pitch-distribution baselines at 12–1200 bins per octave and a reproduction of TDMS (Gulati et al., 2016)
- 84 per-svara **shruti descriptors** (intonation offset, spread, skew, dwell, gamaka energy)
- **SAR-Net**, a shruti-aware CNN–BiGRU network with attention (about 0.23 M parameters)
- a **pitch-correction simulation** (strength α = 0…1) and **correction-aware training**

## Data

The experiments use the open *Indian Art Music Raga Recognition Dataset (features)*,
Gulati et al. (2016), Zenodo: <https://doi.org/10.5281/zenodo.7278506> (3.6 GB zip; 480 Carnatic
recordings / 40 rāgas, 300 Hindustani recordings / 30 rāgas). Audio is not needed.

1. Download the zip and place it next to `data_prep/prep_raga_features.py`.
2. Run `python data_prep/prep_raga_features.py` (on Windows, double-click `data_prep/run_prep.bat`).
   It reads the zip directly and writes `raga_compact.npz` (~24 MB) in about 8 minutes.
3. Copy that file to `data/raga.npz`.

## Running the experiments

```bash
pip install -r requirements.txt
cd src
python classical.py ../data/raga.npz ../data/classical.json   # baselines, shruti descriptors, robustness (~5 min)
python deep.py      ../data/raga.npz ../data/deep.json both 20 # SAR-Net, 3 variants x 2 traditions (~60 min on 2 cores)
python extra.py                                                 # timings, confusion matrix, rank fusion, McNemar
python figs_classical.py                                        # Figures 2-4 of the paper
cd .. && python src/fig_architecture.py                         # Figure 1 (architecture)
```

All results reported in the paper are already included in `data/*.json`, and the figures in `figs/`.

## Main results (five-fold cross-validation)

| System | Carnatic (40 rāgas) | Hindustani (30 rāgas) |
|---|---|---|
| TDMS + 1-NN (reproduction) | 84.6% | 96.8% |
| PD-120 + shruti descriptors + SVM | 83.2% | 96.6% |
| SAR-Net | 84.2% | 95.7% |
| SAR-Net + TDMS (rank fusion) | 84.8% | 97.3% |
| Correction-aware SAR-Net, any correction strength | 82.7–85.0% | 93.7–96.7% |

## Repository layout

```
data_prep/   compaction script for the Zenodo zip (+ Windows launcher)
src/         features, classical experiments, SAR-Net, analysis and figure scripts
data/        result files (JSON) produced by the scripts
figs/        figures used in the manuscript
```

## Citation

If you use this code, please cite the manuscript (citation details will be added on publication) and the dataset:
Gulati, S., Serrà, J., Ganguli, K. K., Şentürk, S., & Serra, X. (2016). Time-delayed melody surfaces for rāga
recognition. *Proceedings of the 17th ISMIR Conference*, 751–757.

## License

MIT, see `LICENSE`. The dataset is distributed by its creators under its own terms.
