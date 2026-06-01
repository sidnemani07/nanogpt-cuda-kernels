# Dev Log

> One sentence per day. What did I work on, what blocked me, what's tomorrow.

Future-me reads this on bad days. BIShIP application readers might too.

---

## Week 0 — Setup

**2026-05-31 (Day 0)**: Initialized repo. Wrote README with full 10-week plan and daily schedule. Free GPU setup pending (Colab + Kaggle).

---

## Week 1 — CUDA Fundamentals (June 1 – June 7)

**Goal**: GPU MODE lectures 1-3 done. Three working kernels: vector_add, transpose, naive_matmul. Each with a test and benchmark.

**2026-06-01 (Day 1)**:Shipped vector_add kernel (out[i]=a[i]+b[i]) + standalone test/benchmark. Correct vs torch.add at 1K/1M/10M. T4 hits 264 GB/s sustained @ 50M = 82% of ~320 peak. Lesson: memory-bound op, naive ≈ optimal — no trick beats the bandwidth wall. Small n (75%) hides true throughput because launch overhead dominates. Tomorrow: transpose.
**2026-06-02 (Day 2)**:
**2026-06-03 (Day 3)**:
**2026-06-04 (Day 4)**:
**2026-06-05 (Day 5)**:
**2026-06-06 (Day 6)**:
**2026-06-07 (Day 7)**:

**Week 1 retro**:
- Shipped:
- Learned:
- Surprised me:
- Blocking week 2:

---

## Week 2 — Reduction & softmax (June 8 – June 14)

**2026-06-08**:
**2026-06-09**:
**2026-06-10**:
**2026-06-11**:
**2026-06-12**:
**2026-06-13**:
**2026-06-14**:

**Retro**:

---

## Week 3 — Karpathy + mini-transformer (June 15 – June 21)

**2026-06-15**:
**2026-06-16**:
**2026-06-17**:
**2026-06-18**:
**2026-06-19**:
**2026-06-20**:
**2026-06-21**:

**Retro**:

---

## Week 4 — Train GPT-2 tiny (June 22 – June 28)

**2026-06-22**:
**2026-06-23**:
**2026-06-24**:
**2026-06-25**:
**2026-06-26**:
**2026-06-27**:
**2026-06-28**:

**Retro**:

---

## Week 5 — Fused LayerNorm (June 29 – July 5)

**2026-06-29**:
**2026-06-30**:
**2026-07-01**:
**2026-07-02**:
**2026-07-03**:
**2026-07-04**:
**2026-07-05**:

**Retro**:

---

## Week 6 — Fused GeLU + integration (July 6 – July 12)

**2026-07-06**:
**2026-07-07**:
**2026-07-08**:
**2026-07-09**:
**2026-07-10**:
**2026-07-11**:
**2026-07-12**:

**Retro**:

---

## Week 7 — Flash Attention forward (July 13 – July 19)

**2026-07-13**:
**2026-07-14**:
**2026-07-15**:
**2026-07-16**:
**2026-07-17**:
**2026-07-18**:
**2026-07-19**:

**Retro**:

---

## Week 8 — Flash Attention backward (July 20 – July 26)

> THE HARDEST WEEK. Block out distractions.

**2026-07-20**:
**2026-07-21**:
**2026-07-22**:
**2026-07-23**:
**2026-07-24**:
**2026-07-25**:
**2026-07-26**:

**Retro**:

---

## Week 9 — Integration + final benchmarks (July 27 – August 2)

**2026-07-27**:
**2026-07-28**:
**2026-07-29**:
**2026-07-30**:
**2026-07-31**:
**2026-08-01**:
**2026-08-02**:

**Retro**:

---

## Week 10 — Writeup + polish (August 3 – August 9)

**2026-08-03**:
**2026-08-04**:
**2026-08-05**:
**2026-08-06**:
**2026-08-07**:
**2026-08-08**:
**2026-08-09**:

**Project retro**:
- Final count of what shipped:
- Biggest lessons:
- What I'd do differently:
- What this unlocks for BIShIP applications:
- Next project (Fall 2026):