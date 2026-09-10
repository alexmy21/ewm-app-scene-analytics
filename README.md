# ewm-app-scene-analytics

An application-level demo project of the **ewm-app** line: scene-change
detection in a short video clip, computed in two spaces and compared, then
analyzed with lattice-native methods that only HLLSets make possible.

```text
                 ┌──────────────────────────────┐
                 │  frame i  (568x320, 100 of)  │
                 └──────────────┬───────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
   ┌────────────────────┐                  ┌──────────────────────┐
   │ vLLM line          │                  │ HLLSet lattice line  │
   │ DeepSeek-OCR       │                  │ (gen2 foundation)    │
   │ CLIP-L vision tower│                  │ patch → tid tokens   │
   │ 256 patches x 1024 │                  │ → HLLSet::from_tokens│
   └─────────┬──────────┘                  └─────────┬────────────┘
             │                                       │
             ▼                                       ▼
   centered cosine similarity              BSSτ = |A∩B| / |B|
   mapped to [0,1]                         (HLLSet lattice, [0,1])
             │                                       │
             └───────────────┬───────────────────────┘
                             ▼
            similarity series over consecutive frames
            lowest points  ⇒  scene changes
```

## What the notebook does

`notebooks/01_vllm_scene_change_hllset.ipynb` (executed, figures embedded)
runs five layers over a 10-scene, 100-frame clip:

1. **vLLM line** — raw encodings from the DeepSeek-OCR CLIP-L vision tower
   (256 patches × 1024-dim per frame), centered cosine similarity mapped to
   `[0,1]`. Pearson 0.94 with the lattice line; detects the strongest cuts.
2. **HLLSet lattice line** — the same patches quantized to `tid{n}` tokens
   and inscribed as one HLLSet per frame through the gen2 `hllset` CLI;
   consecutive frames compared with BSSτ. Detects **9/9** cuts.
3. **Moving-average model** — the stock-trading trick in HLLSet space:
   n-gram trailing unions of HLLSets as moving averages. The fast 1-gram line
   crosses below the slow 5-gram MA at every scene cut. Detects **9/9** cuts
   with zero false positives.
4. **Noether D/R/N multi-expert model** — `H(t) = (S(t), H(t-1), D, R, N)`
   per frame transition (D dropped, R retained, N new, all HLLSet set
   algebra). Three experts — `|D|/|R∪N|`, `1 − BSS(R_t,R_{t-1})`,
   `|N|/|R∪D|` — are ranked by separation accuracy and weighted into one
   decision line (z-scores). Detects **9/9** cuts.
5. **Real-footage run** — the same pipeline on a continuous-motion cat clip,
   as a sanity check outside the synthetic ground truth.

## Layout

```text
ewm-app-scene-analytics/
├── README.md
└── notebooks/
    └── 01_vllm_scene_change_hllset.ipynb   # the application
```

## How to run

The notebook is a **Python** notebook (the gen2 `hllset-next-v2` notebooks
01–06 are Rust/evcxr; this application notebook needs PyTorch + matplotlib +
the local vision model, so it lives here in its own project).

Requirements on this laptop:

- conda env `deepseek-ocr` (Python 3.10, torch 2.4, vllm 0.6.3,
  transformers 4.46.3) — the **vLLM line** (DeepSeek-OCR's CLIP-L vision
  tower, loaded from the local HuggingFace cache);
- the Jupyter kernel `deepseek-ocr` (install once with
  `~/.conda/envs/deepseek-ocr/bin/python -m ipykernel install --user
  --name deepseek-ocr --display-name "Python (deepseek-ocr)"`);
- the gen2 foundation `hllset-next-v2` checked out next to this project
  (the notebook builds `hllset-cli` on first use and calls it for the
  **HLLSet lattice line**).

```bash
jupyter notebook notebooks/01_vllm_scene_change_hllset.ipynb   # pick the deepseek-ocr kernel
```

Dependency direction follows GOVERNANCE.md §5: this application project
consumes the foundation (`hllset-next-v2`) and the local model, and never
modifies them.

## Status

Executed green 2026-09-09 on the laptop (RTX 3060):

| analysis layer | scene cuts detected |
| --- | --- |
| vLLM line (centered cosine) | 6/9 (strongest cuts) |
| HLLSet lattice line (BSSτ) | 9/9 |
| Moving-average cross (1-gram vs 5-gram) | 9/9, zero false positives |
| Noether D/R/N experts (weighted decision) | 9/9 |

See the notebook for the full numbers and diagrams.

## Next directions

HLLSet is a set by behaviour, so this project can grow into the full
statistical toolbox without leaving the lattice:

- **Bayesian analysis** — BSSτ is already a conditional measure
  (`|A∩B|/|B|`); `inscribe` = likelihood, `materialize` = posterior,
  union/intersection = evidence combination.
- **Markov chains** — content-addressed keys are the states; the Noether
  D/R/N counts are the empirical transition matrices.
- **Metrics and graphs** — Jaccard/BSS metric spaces (clustering, kernels),
  entropy over popcounts, bitwise OR/AND as the Boolean-lattice basis.
