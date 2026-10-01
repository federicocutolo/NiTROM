# Examples

Each directory under `examples/` in the repository is self-contained: generate data,
train, then post-process.

The **toy model** is executed as a live gallery when these docs are built — every
figure you see there was produced by the committed scripts:

```{toctree}
:maxdepth: 2

../auto_examples/index
```

The larger fluid-dynamics cases need substantial data generation and compute, so they
are documented as static pages:

```{toctree}
:maxdepth: 1

cavity
airfoil
```
