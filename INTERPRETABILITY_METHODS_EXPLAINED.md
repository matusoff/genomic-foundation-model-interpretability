# Interpretability Methods Explained

## 1. Position Ablation Analysis

**What it does:** Tests which **DNA sequence positions** are most important for the model's predictions.

**How it works:**
1. **Baseline:** Run the model on the full sequence → get output
2. **Ablate:** Replace specific positions with `<mask>` token → run model again
3. **Compare:** Measure how much the output changed (impact)

**Details:**
- **Window:** Tests positions ±100 nucleotides around the variant (positions 156-356 if variant is at 256)
- **Step size:** Tests every 5th position (so ~40 positions total)
- **Ablation size:** At each test position, masks **5 consecutive nucleotides** (positions `pos` to `pos+5`)
- **Token mapping:** Since the model uses 6-mer tokenization, ~6 nucleotides = 1 token, so masking 5 nucleotides affects ~1 token

**Example:**
```
Sequence: ...ATCGATCGATCGATCG...
           ^^^^^^
           Position 200-210 (10 nucleotides masked)
```

**What the plot shows:**
- **Top plot:** Impact when ablating each position (blue=reference, red=alternate)
- **Bottom plot:** Difference between ref and alt impacts
- **Higher impact** = that position is more critical for the model

**In results:**
- **40 positions tested** (200 nt window ÷ 5 step size)
- Each test ablates **5 consecutive nucleotides**
- Shows which regions around the variant matter most

---

## 2. Circuit Analysis (Attention Head Ablation)

**What it does:** Tests which **internal model components** (attention heads in specific layers) are most important.

**How it works:**
1. **Baseline:** Run model normally - get final hidden states
2. **Ablate:** Zero out a specific attention head's output - run model again
3. **Compare:** Measure how much final output changed

**Details:**
- **What's ablated:** One attention head at a time (e.g., Layer 12, Head 4)
- **How:** The head's contribution is set to zero in the hidden states
- **Sampling:** Tests last 10 layers (12-21) × 8 heads (0,2,4,6,8,10,12,14) = **80 combinations**
- **Full model:** 22 layers × 16 heads = 352 combinations (we sample for speed)

**Example:**
```
Layer 12, Head 4:
  Before: [hidden states with all heads active]
  After:  [hidden states with Head 4 = 0, others active]
```

**What the plot shows:**
- **Heatmap:** Impact difference for each layer-head combination
- **Top circuits:** Which specific heads matter most
- **By layer:** Deeper layers (higher numbers) usually more important
- **By head:** Some heads specialize in different features

**In your results:**
- 80 layer-head combinations tested
- Shows which internal circuits differentiate ref vs alt sequences

---

## 3. Attention Analysis

**What it does:** Shows **where the model "looks"** when processing the sequence.

**How it works:**
1. **Extract:** Get attention weights from all layers/heads
2. **Compare:** Attention patterns for reference vs alternate sequences
3. **Visualize:** Heatmaps showing attention differences

**Details:**
- **Attention weights:** For each position, shows how much it "attends to" other positions
- **Shape:** [22 layers, 16 heads, 512 positions, 512 positions]
- **Meaning:** High attention = model connects those positions
- **Variant focus:** Checks if model pays more attention to variant position in alt vs ref

**What the plot shows:**
- **Heatmap:** Attention difference (Alt - Ref) for one layer-head
- **X-axis:** Query position (where attention comes from)
- **Y-axis:** Key position (where attention goes to)
- **Color:** Red = more attention in alt, Blue = more attention in ref
- **Yellow lines:** Variant position (should show pattern here if model cares about it)

**Example:**
```
Position 250 (variant):
  Ref:  Low attention to position 250
  Alt:  High attention to position 250
  → Model notices the variant in alt sequence
```

**In results:**
- Multiple heatmaps (one per layer-head combination shown)
- Shows if model's "focus" changes between ref and alt sequences

---

## Summary Table

| Method | What's Changed | How Many Tests | What It Shows |
|--------|---------------|----------------|---------------|
| **Position Ablation** | Mask 5 nucleotides at a time | 40 positions | Which sequence regions matter |
| **Circuit Analysis** | Zero out 1 attention head | 80 layer-head combos | Which internal components matter |
| **Attention Analysis** | Nothing (just observe) | All layers/heads | Where model "looks" in sequence |

---


