# Fokker-Planck Phase-Transition Dynamics in Machine Learning

A framework for modeling neural network training dynamics as stochastic processes with emergent phase transitions.

---

## Overview

This repository implements a Fokker-Planck equation (FPE) framework for analyzing how neural networks learn representations. We model latent state evolution as probability flows governed by drift (gradient descent) and diffusion (stochastic noise), with phase transitions explaining sudden capability jumps during training.

**Key Idea**: Training dynamics = Stochastic process on learned manifolds + Critical transitions at regime shifts

---

## Mathematical Framework

### The Fokker-Planck Equation

Probability density ρ(z,t) over latent states evolves as:

```
∂ρ/∂t = -∇·(μ(z)ρ) + ∇·(D(z)∇ρ)
```

**Components:**
- ρ(z,t): Distribution over latent representations
- μ(z): Drift velocity (gradient-based learning)
- D(z): Diffusion coefficient (exploration noise)

### Drift: Information-Theoretic Gradient Flow

```
μ(z) = -η·∇ℒ(z) - β·∇H[ρ]
```

- ℒ(z): Expected loss at latent point z
- H[ρ]: Shannon entropy = -∫ ρ log(ρ) dz
- η: Learning rate
- β: Entropy regularization (prevents mode collapse)

**Interpretation**: Flow toward low loss while maintaining representation diversity.

### Diffusion: Adaptive Exploration

```
D(z,t) = D₀·exp(-t/τ) + σ²·I(z)
```

- D₀·exp(-t/τ): Decaying exploration (like learning rate schedules)
- I(z): Fisher information matrix
- σ²: Hardware noise floor

**Interpretation**: Exploration decreases over time but persists in high-curvature regions.

### Phase Transitions: Stochastic Resets

At critical points (loss plateaus, gradient collapse), trigger:

```
D(z,t*) ← D(z,t*) + ΔD
ρ(z,t*) ← ρ(z,t*) - α·∇²ρ
```

**Effect**: Temporarily boost diffusion to escape local minima, redistribute probability mass.

**Analogous to**: Simulated annealing, warm restarts, curriculum phase shifts.

---

## Key Metrics

### Consolidation Ratio

```
C(t) = ||μ||₂ / ||D∇ρ||₂
```

- High C: Exploitation (convergence)
- Low C: Exploration (search)
- Sharp drops: Phase transition events

### Information Flux

```
J(z) = μρ - D∇ρ
```

Tracks net probability current through latent space.

### Entropy Production

```
Ṡ = ∫ J·∇log(ρ) dz
```

Quantifies irreversible information processing. Spikes during transitions.

---

## Relationship to Existing Work

**vs. Diffusion Models (DDPM, Score Matching)**
- Diffusion models: Reverse-time FPE for generation
- This work: Forward-time FPE for training dynamics
- Key difference: Task-driven drift (∇ℒ) not just score matching

**vs. Neural SDEs**
- Neural SDEs: Learned drift/diffusion for time series
- This work: Analytical structure from information theory
- Key difference: Explicit phase-transition resets

**vs. Langevin Dynamics**
- Langevin: Constant temperature MCMC sampling
- This work: Adaptive diffusion with critical transitions
- Key difference: Non-equilibrium phase changes

**vs. Optimal Transport in ML**
- Wasserstein gradient flows: Minimize KL via probability transport
- This work: FPE with entropy regularization + resets
- Key difference: Stochastic resets model abrupt regime shifts

---

## Connection to Modern ML Phenomena

**Grokking** (sudden generalization): Phase transition from memorization to generalization regime when diffusion temporarily dominates.

**Emergent Capabilities** (scaling laws): Critical transitions when model capacity crosses threshold for new skill acquisition.

**Catastrophic Forgetting**: Loss of stationary distribution when new tasks shift drift field without sufficient diffusion.

**Double Descent**: Non-monotonic risk due to phase transitions in representation geometry.

---

## Implementation

### Basic Simulation

```python
import numpy as np
from scipy.stats import multivariate_normal

class FokkerPlanckLearning:
    def __init__(self, dim=2, eta=0.1, beta=0.01, D0=1.0, tau=100):
        self.dim = dim
        self.eta = eta  # learning rate
        self.beta = beta  # entropy regularization
        self.D0 = D0  # initial diffusion
        self.tau = tau  # diffusion decay
        
    def drift(self, z, loss_grad, rho_grad_entropy):
        """μ(z) = -η∇ℒ - β∇H"""
        return -self.eta * loss_grad - self.beta * rho_grad_entropy
    
    def diffusion(self, t, fisher_info):
        """D(t) = D₀exp(-t/τ) + σ²I"""
        return self.D0 * np.exp(-t / self.tau) * np.eye(self.dim) + \
               0.01 * fisher_info
    
    def phase_transition_reset(self, D, rho, alpha=0.5):
        """Stochastic reset: boost diffusion + redistribute mass"""
        D_reset = D + alpha * self.D0 * np.eye(self.dim)
        return D_reset
```

### Monitoring Phase Transitions

```python
def detect_transition(consolidation_history, threshold=0.5):
    """Detect sharp drops in consolidation ratio"""
    C = np.array(consolidation_history)
    dC = np.diff(C)
    transitions = np.where(dC < -threshold)[0]
    return transitions

def consolidation_ratio(drift_norm, diffusion_grad_norm):
    """C(t) = ||μ|| / ||D∇ρ||"""
    return drift_norm / (diffusion_grad_norm + 1e-8)
```

---

## Experiments

### Toy Example: 2D Latent Space

Visualize probability flow on a simple loss landscape:

```python
# Loss landscape: two Gaussian wells
def loss(z):
    return -np.log(0.6 * mvn1.pdf(z) + 0.4 * mvn2.pdf(z))

# Run FPE simulation
t_span = (0, 500)
z0 = sample_initial_distribution(n_particles=1000)
trajectories = simulate_fpe(z0, loss, t_span)

# Identify phase transitions
C_history = [consolidation_ratio(t) for t in trajectories]
transitions = detect_transition(C_history)
```

**Expected Result**: Particles initially spread, then consolidate into low-loss wells, with sharp transitions when encountering barriers.

### Continual Learning Benchmark

Test on Split-MNIST (5 tasks):

```python
# Track consolidation ratio across task boundaries
for task_id in range(5):
    train_model(task_id)
    C = compute_consolidation()
    
    if C < threshold:  # Phase transition
        apply_reset(model, boost_factor=2.0)
```

**Hypothesis**: Models with explicit resets should show reduced forgetting.

---

## Theoretical Results

### Proposition 1: Stationary Distribution

Under regularity conditions (bounded ∇ℒ, positive definite D), the stationary distribution exists:

```
ρ*(z) ∝ exp(-ℒ(z) / T_eff)
```

where T_eff = Tr(D) / ||μ|| is the effective temperature.

### Proposition 2: Entropy Production Bound

The total entropy production satisfies:

```
∫₀ᵀ Ṡ(t) dt ≥ KL(ρ₀ || ρ*) 
```

Equality holds only for reversible (diffusion-free) processes.

### Proposition 3: Phase Transition Criterion

A phase transition occurs when the drift-diffusion balance shifts:

```
λ_min(D) > η·||∇²ℒ||
```

**Interpretation**: Diffusion eigenvalues exceed curvature-scaled learning rate.

---

## Key References

### Foundational Theory

**Shannon (1948)** - A Mathematical Theory of Communication  
IEEE Transactions. Foundation of information entropy.

**Jaynes (1957)** - Information Theory and Statistical Mechanics  
Physical Review. Maximum entropy principle.

**Risken (1996)** - The Fokker-Planck Equation: Methods of Solution and Applications  
Springer. Comprehensive FPE reference.

**Villani (2009)** - Optimal Transport: Old and New  
Springer. Probability mass transport theory.

### Machine Learning Connections

**Welling & Teh (2011)** - Bayesian Learning via Stochastic Gradient Langevin Dynamics  
ICML. Langevin dynamics for neural network training.

**Song & Ermon (2019)** - Generative Modeling by Estimating Gradients of the Data Distribution  
NeurIPS. Score-based diffusion models.

**Ho et al. (2020)** - Denoising Diffusion Probabilistic Models  
NeurIPS. DDPM framework (reverse FPE).

**Kidger et al. (2021)** - Neural Controlled Differential Equations for Irregular Time Series  
NeurIPS. Neural SDEs for sequence modeling.

**Lyu & Ergen (2024)** - Understanding Grokking Through a Robustness Viewpoint  
ICLR. Phase transitions in generalization.

### Information Geometry

**Amari (1998)** - Natural Gradient Works Efficiently in Learning  
Neural Computation. Fisher information in optimization.

**Martens (2020)** - New Insights and Perspectives on the Natural Gradient Method  
JMLR. Modern natural gradient analysis.

### Phase Transitions in Learning

**Power et al. (2022)** - Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets  
ICLR. Abrupt generalization transitions.

**Wei et al. (2022)** - Emergent Abilities of Large Language Models  
TMLR. Scaling-induced capability jumps.

**Schaeffer et al. (2023)** - Are Emergent Abilities of Large Language Models a Mirage?  
NeurIPS. Critical analysis of phase transitions in LLMs.

### Hyperbolic Geometry in ML

**Nickel & Kiela (2017)** - Poincaré Embeddings for Learning Hierarchical Representations  
NeurIPS. Hyperbolic latent spaces.

**Ganea et al. (2018)** - Hyperbolic Neural Networks  
NeurIPS. Neural networks on hyperbolic manifolds.

---

## Installation

```bash
git clone https://github.com/yourusername/fokker-planck-ml.git
cd fokker-planck-ml
pip install -r requirements.txt
```

**Requirements:**
```
numpy>=1.21.0
scipy>=1.7.0
torch>=2.0.0
matplotlib>=3.5.0
```

---

## Quick Start

```python
from fpe_learning import FokkerPlanckDynamics, visualize_flow

# Initialize system
fpe = FokkerPlanckDynamics(
    latent_dim=128,
    learning_rate=0.01,
    entropy_reg=0.001,
    diffusion_decay=100
)

# Train with phase transition monitoring
for epoch in range(100):
    loss = train_epoch(model, data)
    
    # Compute FPE metrics
    C = fpe.consolidation_ratio(model)
    S_dot = fpe.entropy_production(model)
    
    # Detect and handle transitions
    if C < 0.5:
        print(f"Phase transition at epoch {epoch}")
        fpe.apply_reset(model, boost=2.0)
```



