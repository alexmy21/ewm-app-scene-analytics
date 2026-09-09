# ewm-app-scene-analytics

An application-level demo project of the **ewm-app** line: scene-change
detection in a short video clip, computed in two spaces and compared.

```text
                 ┌──────────────────────────────┐
                 │  frame i  (568x320, 100 of)  │
                 └──────────────┬───────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
   ┌────────────────────┐                  ┌────────────────────┐
   │ vLLM line          │                  │ HLLSet lattice line │
   │ DeepSeek-OCR       │                  │ (gen2 foundation)   │
   │ CLIP-L vision tower│                  │ patch → tid tokens  │
   │ 256 patches x 1024 │                  │ → HLLSet::from_tokens│
   └─────────┬──────────┘                  └─────────┬──────────┘
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

The demo then adds the **moving-average model** — the stock-trading trick,
computed with n-gram unions of HLLSets instead of floating-point averages:
the fast 1-gram line crosses below the slow 5-gram HLLSet moving average at
every scene cut.

## Layout

```text
ewm-app-scene-analytics/
└── notebooks/
    └── 01_vllm_scene_change_hllset.ipynb   # the application (executed, figures embedded)
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

Executed green 2026-09-09 on the laptop (RTX 3060): both lines detect the
scene cuts; the moving-average cross detects 9/9 cuts with zero false
positives. See the notebook for the numbers and diagrams.
