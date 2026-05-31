# Flash Attention — Intuition (read this first in Week 7)

> Written Day 0 (2026-05-31) while the idea was fresh. Future-me: you understood this
> once with zero CUDA under your belt. The concept is easy. The CUDA is the grind.
> This file is here so you don't have to re-learn the *why* — just the *how*.

---

## The one sentence

An equation can't make data faster. But for the attention equation, there are slow
ways and fast ways to run it on a GPU, and the difference is almost entirely **how much
data you drag through slow memory.** Flash Attention is the fast way. Same answer.

---

## The equation it computes

```
O = softmax( (Q Kᵀ) / √d ) V
```

Read as a sentence: for every word, **match** its question (Q) against every word's
label (K) → `QKᵀ`. **Calm** the numbers by dividing by √d. Turn them into **attention
percentages** with softmax. Then build each word's new meaning as a **weighted mix** of
everyone's content (V).

- **Q (Query)** = "what I'm looking for"
- **K (Key)** = "what I advertise about myself"
- **V (Value)** = "what I actually give you if you pick me"

Flash Attention computes this EXACT equation. It is not different math. It is not an
approximation. It is the same `O`, run smarter.

---

## Why naive attention is slow

A GPU has two memories:
- **SRAM** — tiny, on-chip, BLAZING fast (the kitchen counter)
- **HBM** — huge, main GPU memory, SLOW (the fridge)

The bottleneck on a GPU is **moving data, not doing math.** The math units finish and
sit bored, waiting for data to arrive from slow HBM.

The naive way builds the whole `N × N` score matrix (for 4000 tokens that's 16 MILLION
numbers), writes it to slow HBM, reads it back for softmax, writes again, reads again to
multiply by V. The giant matrix gets dragged through slow memory ~4 times. That dragging
is the slow part — not the arithmetic.

---

## How Flash Attention gets the data faster — THE THREE MOVES

### Move 1 — TILING
Don't build the giant scoreboard. Grab a small **block** of Q and a small block of K/V,
small enough to fit in fast SRAM. Compute that little corner right there, use it, throw
it away, grab the next block. The giant matrix never exists all at once, so it never has
to be parked in slow HBM. (Same tiling as Week 1 matmul.)

### Move 2 — ONLINE SOFTMAX (the clever bit)
Problem: softmax needs the **whole row** to make percentages (it needs the row's sum).
But tiling only shows you part of the row at a time. Fix: keep a **running tally** —
carry two numbers per row (running max + running sum). Process a tile → partial answer.
New tile arrives → rescale the running answer, then add the new contribution. By the
last tile, the result is IDENTICAL to having seen the whole row at once. This is where
the Week 8 bugs live (getting the rescaling numerically exact).

### Move 3 — FUSION
The naive way runs matmul → (HBM) → softmax → (HBM) → ×V → (HBM): ~6 fridge trips.
Flash fuses all of it into ONE kernel: while a tile is hot in SRAM, do matmul + softmax
math + ×V, and only write the final small result out. 1 fridge trip. (Weeks 5–6 fused
LayerNorm / GeLU are practice for this.)

### The loop
```
For each block of the sequence:
   1. TILING          → load a small Q/K/V tile into fast SRAM
   2. FUSION          → do matmul + softmax-math + ×V right there, hot
   3. ONLINE SOFTMAX  → update running tally so partials = exact final answer
Write only the final result to slow memory. Done.
```

Result: memory traffic drops from "drag 16M numbers through slow memory several times"
to "stream small tiles through fast memory once." Same `O`. Far fewer trips. That's the
whole win. Memory also drops from O(N²) to O(N).

---

## Why this is the capstone, not the start

Flash Attention is not a new thing to learn — it's the three things you're already
learning, stacked:

- **Tiling** → Week 1 (matmul)
- **Online softmax** → Week 2 (softmax), upgraded to the running-tally version
- **Fusion** → Weeks 5–6 (fused LayerNorm + GeLU)

By Week 7 each move is muscle memory. Week 7 is just combining reflexes you already own.
The forward pass (Week 7) = tiling + online softmax + fusion. The backward pass (Week 8,
"THE HARDEST WEEK") = recompute the tiles during the gradient pass instead of storing
them.

---

## Diagrams

- `flash_attention_explained.png` — naive vs flash, the fridge-trips picture
- `flash_attention_three_moves.png` — tiling / online softmax / fusion

## Source to read in Week 7
- FlashAttention paper (Dao et al., 2022): https://arxiv.org/abs/2205.14135
