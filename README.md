# VisNet-Unified-Framework

A reproducible framework for local plasticity in hierarchical visual learning, integrating multi-frequency streams, dorsal pathways, and associative memory for invariant representation learning.

---

## Motivation

Classical biologically inspired vision models such as VisNet demonstrate that invariant object recognition can emerge from local learning rules and temporal continuity. However, these models typically isolate individual mechanisms (e.g., trace learning or competition) and do not provide a unified platform to study how multiple biologically motivated processes interact.

This repository introduces a **unified hierarchical framework** that integrates multiple mechanisms observed in cortical processing—such as multi-scale feature extraction, lateral competition, predictive coding, and associative memory—within a single modular system.

The goal is not only to improve representation quality, but to provide a **controlled experimental platform** for evaluating how combinations of local plasticity rules contribute to invariant visual representations.

---

## Framework Variants / Configurations

visnet-unified-full — complete framework with all components enabled (multi-frequency streams, dorsal pathway, PFC memory, top-down refinement)

visnet-unified-hebbian — minimal configuration using only Hebbian/Oja learning (baseline)

visnet-unified-no-pfc — removes associative memory and top-down modulation

visnet-unified-no-topdown — disables iterative feedback refinement

visnet-unified-single-stream — reduces multi-frequency processing to a single stream

---

## visnet-unified-full: Complete Hierarchical Framework

The full model integrates multiple biologically inspired components into a single architecture:

* **Multi-frequency ventral streams** simulate parallel processing across spatial scales (similar to V1–V4 frequency selectivity)
* **Topographic hierarchy** preserves spatial organization across layers
* **Lateral inhibition** introduces competition and decorrelation
* **Predictive coding signals** provide error-driven modulation via reconstruction pathways
* **Temporal trace learning** promotes invariance across transformations
* **Prefrontal Cortex (PFC) memory** implements associative retrieval using a modern Hopfield mechanism
* **Top-down modulation** enables iterative refinement of representations

Unlike traditional models, this framework is designed as a **compositional system**, where each component can be independently enabled, disabled, and evaluated.

---

## visnet-unified-hebbian: Minimal Local Learning Baseline

This configuration isolates the core Hebbian/Oja learning mechanism by disabling all auxiliary components (trace, predictive coding, PFC, dorsal pathway).

From a scientific perspective, this serves as a **baseline reference model**, allowing researchers to measure the contribution of additional mechanisms relative to pure local correlation-based learning.

Without temporal or top-down signals, the model learns static feature representations driven solely by input statistics. This configuration is critical for understanding how much structure can emerge from simple local rules alone.

---

## visnet-unified-no-pfc: Without Associative Memory

This variant removes the PFC module and all top-down retrieval mechanisms.

The architecture retains feedforward and lateral processing, but loses:

* Global context integration
* Memory-based retrieval
* Iterative refinement

This allows evaluation of how **associative memory contributes to representation quality**, particularly in ambiguous or complex visual inputs.

---

## visnet-unified-no-topdown: Feedforward + Lateral Only

This configuration disables iterative top-down refinement while keeping the PFC module inactive during inference.

The model becomes a **purely feedforward + lateral system**, similar to classical hierarchical vision models.

This variant is important for isolating the effect of **feedback loops and recurrent processing**, which are hypothesized to play a key role in biological vision.

---

## visnet-unified-single-stream: Reduced Frequency Processing

In this variant, the multi-frequency architecture is collapsed into a single stream.

This removes:

* Multi-scale feature decomposition
* Frequency diversity

The comparison between this and the full model quantifies the importance of **multi-scale representations** in invariant learning.

---

## Installation

```bash
git clone https://github.com/mehdifatan/VisNet-Unified-Framework.git
cd VisNet-Unified-Framework
pip install -r requirements.txt
```

---

## Usage

### Full model

```bash
python VisNet_Unified_Framework_ToOne12_PFC23_Ventral_Dorsal21.py \
  --epochs 300 --batch-size 4 --lr 1e-3 \
  --wavelet-input --pfc-mode hopfield \
  --pfc-topdown-iters 2 \
  --dorsal-enabled True
```

---

### Example Ablations

**Hebbian-only**

```bash
--lambda-fe 0.0 --alpha-trace 0.0 \
--no-use-pfc-hopfield --pfc-topdown-iters 0
```

**No top-down**

```bash
--pfc-topdown-iters 0
```

**Single stream**

```bash
--num-gabor-freqs 1
```

---

## References

Rolls, E. T. (2012). Invariant visual object and face recognition: Neural and computational bases, and a model, VisNet.

Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex.

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities.

Ramsauer, H. et al. (2020). Hopfield networks is all you need.

Hebb, D. O. (1949). The Organization of Behavior.

Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields in visual cortex.

---

## Roadmap

* Complete ablation studies for all components
* Add baseline comparisons (SimCLR, BYOL, CNNs)
* Extend to continual learning (EWC, PackNet)
* Evaluate on video datasets (UCF-101, Kinetics)
* Implement sparse lateral connections for scaling

---

## Citation

If you use this framework, please cite:

```bibtex
@article{fatan2026visnet_unified,
  title   = {A Unified Framework for Studying Local Learning in Hierarchical Visual Representations},
  author  = {Fatan Serj, Mehdi and Parraga, C. Alejandro and Otazu, Xavier},
  year    = {2026}
}
```

And optionally the repository:

```bibtex
@misc{fatan2026visnet_unified_repo,
  author = {Mehdi Fatan Serj},
  title  = {VisNet-Unified-Framework},
  year   = {2026},
  howpublished = {\url{https://github.com/mehdifatan/VisNet-Unified-Framework}}
}

* Or create a **1-minute “Quick Start” section for users**
* Or optimize this README for **GitHub trending visibility** 🚀
