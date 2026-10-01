# Architecture

NiTROM is a small number of abstract contracts and a training loop that only talks to
those contracts. Everything else — GAS stability, Hamiltonian structure, polynomial
manifolds — is a subclass slotted into the same pipeline.

## Data flow

```{graphviz}
digraph nitrom_flow {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#4a5568",
          fontname="Helvetica", fontsize=11];
    edge [color="#4a5568", fontname="Helvetica", fontsize=9];

    subgraph cluster_data {
        label="data"; style="rounded"; color="#a0aec0"; fontname="Helvetica"; fontsize=10;
        pool [label="TrainingPool\n(load + shard trajectories)"];
        tdata [label="TrainingData\n(view: subset, quadrature)"];
        pool -> tdata;
    }

    subgraph cluster_rom {
        label="the (projection, model) pair"; style="rounded"; color="#a0aec0"; fontname="Helvetica"; fontsize=10;
        proj [label="Projection\nencode / decode + VJPs"];
        model [label="Model\nlatent RHS, adjoint RHS, VJPs"];
    }

    registry [label="ParamRegistry\none optimization vector,\nshared params summed", fillcolor="#fdf2e3"];
    module [label="InferenceModule\nforward() = cost\ngradient() = analytic grad", fillcolor="#fdf2e3"];
    train [label="train()\nAdam / SGD / L-BFGS\n+ manifold retractions", fillcolor="#e8f5ec"];
    opinf [label="solve_opinf()\nclosed-form least squares", fillcolor="#e8f5ec"];
    rom [label="trained ROM\n(bases + operators)"];

    tdata -> module;
    proj -> registry;
    model -> registry;
    registry -> module;
    module -> train -> rom;
    module -> opinf -> rom;

    backend [label="backend\n(numpy | torch, MPI/torchrun)", shape=component, fillcolor="#f3f0fa"];
    steppers [label="time_steppers\nRK + discrete adjoint,\nsymplectic integrators", shape=component, fillcolor="#f3f0fa"];
    backend -> module [style=dashed, label="serves all layers"];
    steppers -> module [style=dashed];
}
```

A {class}`~nitrom.training_data.TrainingPool` loads and shards trajectories;
a {class}`~nitrom.training_data.TrainingData` view of it is handed to an inference
module together with a *(projection, model)* pair whose parameters a
{class}`~nitrom.roms.ParamRegistry` has flattened into one optimization vector. The
module's `forward()` returns a scalar cost and `gradient()` its analytic gradient;
{func}`~nitrom.optimization.train` (iterative) or
{func}`~nitrom.optimization.solve_opinf` (closed-form, OpInf only) produces the trained
ROM. The array backend and the time steppers are services every layer calls into.

## Which class do I subclass when…

… I want a new **dynamics structure** (a constraint on the assembled operators)
: Subclass {class}`~nitrom.latent_space_models.StructuredPolynomialModel`. You supply
  free parameters, the assembly map onto inner polynomial tensors, and the gradient
  chain rule back; the inner {class}`~nitrom.latent_space_models.PolynomialModel`
  provides RHS, adjoint, and VJPs for free. This is exactly how
  {class}`~nitrom.latent_space_models.GasPolynomialModel` and
  {class}`~nitrom.latent_space_models.HamiltonianPolynomialModel` are built.
  See {doc}`models`.

… I want a new **manifold / encoder–decoder pair**
: Subclass {class}`~nitrom.projections.Projection` (or
  {class}`~nitrom.projections.LinearProjection` if the trial/test bases merely gain a
  constraint, as {class}`~nitrom.projections.VarconLinearProjection` does). You must
  provide `encode`/`decode` and the three VJPs the adjoint needs. See
  {doc}`projections`.

… I want a new **training objective**
: Subclass {class}`~nitrom.optimization.InferenceModule`: implement `forward()`
  returning a scalar cost and `gradient()` returning its analytic gradient per
  parameter. Freezing, manifold typing, and the whole of
  {func}`~nitrom.optimization.train` then work unchanged. See {doc}`training`.

… I want a new **time integrator**
: Add a Butcher tableau in `nitrom.time_steppers.time_stepper` — the Newton stage
  solve, dense output, and the discrete adjoint are written against the tableau, not
  the scheme. The symplectic `implicit_midpoint` and `yoshida4` entered exactly this
  way. See {doc}`backends_steppers`.

```{toctree}
:maxdepth: 1

data
projections
models
training
backends_steppers
```
