# nanoGPT from Scratch with Custom CUDA Kernels

> A summer 2026 project to learn GPU performance engineering by training a small GPT-2 from scratch and replacing its hottest PyTorch operations with hand-written CUDA kernels.

**Author**: Sidharth Nemani — BS Cyber Security Engineering, George Mason University (class of 2029)
**Project window**: June 1 – August 9, 2026 (10 weeks, ~40 hrs/week)
**Hardware**: Google Colab (T4) and Kaggle (P100) free tiers — total project cost $0.

---

## What this project is

By the end of the summer this repo will contain:

1. A working GPT-2 tiny (~10M parameters) trained from scratch on the TinyStories dataset, in pure PyTorch.
2. Three custom CUDA kernels replacing PyTorch operations on the hot path:
   - **Fused LayerNorm** (forward + backward).
   - **Fused GeLU** (forward + backward).
   - **Flash Attention v1** (forward + backward, implemented from the [original paper](https://arxiv.org/abs/2205.14135)).
3. Benchmarks comparing PyTorch baseline vs. each custom kernel, with Nsight Compute profiles.
4. Three technical blog posts explaining what I built and what I learned.

---

## Why this project, why now

The conventional CySE major path doesn't touch GPU performance engineering. There are roughly ~2,000 people in the world who can do production-grade GPU work, and the path to that group runs through projects exactly like this one — paper-implementations, kernel rewrites, end-to-end training. The artifact *is* the resume.

Writing CUDA C++ by hand teaches you the memory hierarchy, occupancy, warp execution, and shared-memory tiling that higher-level GPU abstractions hide from you. Once those are in your hands, Triton and other DSLs become 10x easier later.

---

## The 10-week plan

| Week | Dates | Goal | Deliverable |
|---|---|---|---|
| 1 | Jun 1 – Jun 7 | CUDA fundamentals + first 3 kernels | vector_add, transpose, naive_matmul |
| 2 | Jun 8 – Jun 14 | GPU MODE 4-5 + reduction & softmax | reduction, softmax kernels |
| 3 | Jun 15 – Jun 21 | Karpathy Zero to Hero, build mini-transformer | model.py + tokenizer working |
| 4 | Jun 22 – Jun 28 | Train GPT-2 tiny on Shakespeare → TinyStories | trained checkpoint, loss curve |
| 5 | Jun 29 – Jul 5 | Fused LayerNorm (fwd + bwd) | kernel + tests + benchmark |
| 6 | Jul 6 – Jul 12 | Fused GeLU + integration into training loop | kernel + integration patch |
| 7 | Jul 13 – Jul 19 | Flash Attention forward pass from the paper | forward.cu working |
| 8 | Jul 20 – Jul 26 | Flash Attention backward pass | backward.cu working |
| 9 | Jul 27 – Aug 2 | Integrate Flash Attention, end-to-end benchmark | final integration |
| 10 | Aug 3 – Aug 9 | Write 3 blog posts, polish, ship | blog_drafts/ |

If I fall behind, the priority order is: **finish Flash Attention > finish LayerNorm > finish GeLU**.

---

## Daily structure (8 AM – 5 PM)

| Time | Block |
|---|---|
| 8:00 – 8:30 | Coffee + read CUDA C++ Programming Guide or current paper |
| 8:30 – 11:30 | Deep work block 1: write code |
| 11:30 – 12:00 | Push commits, write a line in DEV_LOG.md |
| 12:00 – 1:00 | Lunch + walk |
| 1:00 – 4:00 | Deep work block 2: debug, test, profile |
| 4:00 – 4:30 | Watch a GPU MODE lecture |
| 4:30 – 5:00 | Plan tomorrow in DEV_LOG.md, commit, push |

---

## Hardware

Free tiers only:

- **[Google Colab](https://colab.research.google.com/)** (free T4 GPU) — primary dev environment
- **[Kaggle Notebooks](https://www.kaggle.com/)** (free P100 GPU, phone-verified) — longer training runs

Total project cost: $0.

---

## Repository layout
.
├── README.md
├── DEV_LOG.md ← daily one-line log
├── LICENSE
├── .gitignore
├── kernels/ ← all custom CUDA kernels
│ ├── week1_fundamentals/
│ ├── week2_basics/
│ ├── week5_layernorm/
│ ├── week6_gelu/
│ └── week7_8_flash_attention/
├── gpt2_tiny/ ← PyTorch GPT-2 implementation
│ ├── model.py
│ ├── train.py
│ ├── data/ ← gitignored
│ └── checkpoints/ ← gitignored
├── benchmarks/ ← timing results
├── profiles/ ← Nsight screenshots
└── blog_drafts/ ← 3 writeups for week 10

text

---

## Resources

**CUDA**: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) · [GPU MODE lectures](https://github.com/gpu-mode/lectures) · [GPU MODE Discord](https://discord.gg/gpumode)

**Transformers**: [Karpathy Zero to Hero](https://karpathy.ai/zero-to-hero.html) · [Karpathy GPT-2 video](https://www.youtube.com/watch?v=l8pRSuU81PU) · [nanoGPT](https://github.com/karpathy/nanoGPT) · [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)

**Flash Attention**: [Original paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) · [Reference implementation](https://github.com/Dao-AILab/flash-attention)

**Profiling**: [Nsight Compute](https://developer.nvidia.com/nsight-compute) · [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

---

## Status

See [DEV_LOG.md](./DEV_LOG.md) for the daily log. Current status: **Week 0** — repo initialized May 31, 2026. Project starts Monday, June 1.

---

## License

MIT.