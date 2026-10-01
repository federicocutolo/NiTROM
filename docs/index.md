# NiTROM

**N**on-**i**ntrusive **T**rajectory-based **R**educed-**O**rder **M**odelling — a Python
package for learning low-dimensional models of high-dimensional dynamical systems
directly from trajectory data.

NiTROM learns a reduced-order model (ROM) as a pair of objects that are trained
**together**:

- a **projection** between the ambient state space $\mathbb{R}^N$ and a latent space
  $\mathbb{R}^r$ (linear/oblique, or a polynomial manifold), and
- a **latent-space dynamics model** $\dot{z} = f(z, u)$ (polynomial, optionally
  constrained to be globally asymptotically stable or Hamiltonian).

Unlike intrusive projection-based ROMs, no access to the full-order operators is
required — only trajectory data. Unlike plain operator inference, the cost is the
*trajectory* mismatch obtained by actually time-marching the ROM, and the bases are
optimized alongside the operators using **analytic adjoint gradients** on matrix
manifolds.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Getting started
:link: getting_started
:link-type: doc
Install the package and train your first ROM in a dozen lines.
:::

:::{grid-item-card} 🧭 Architecture
:link: architecture/index
:link-type: doc
How the layers fit together, and which class to subclass for what.
:::

:::{grid-item-card} 📐 Theory
:link: theory
:link-type: doc
The training objectives, and the Hamiltonian / symplectic machinery.
:::

:::{grid-item-card} 📚 API reference
:link: api/index
:link-type: doc
Every public class and function, organized by architecture layer.
:::

::::

The {doc}`examples <examples_pages/index>` include a fully **executed** toy-model
walkthrough — generate data, train, compare — whose figures are produced live at every
docs build.

## Citing

If you use NiTROM in your research, please cite the NiTROM paper (see the
{doc}`theory` page references).

```{toctree}
:hidden:
:caption: Guide

installation
getting_started
architecture/index
theory
examples_pages/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
```
