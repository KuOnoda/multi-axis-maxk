# Contributing

Use Python 3.10, install `.[test]`, and run `pytest` plus `ruff check .` before
opening a change. Keep the model-independent estimator free of training-stack
dependencies. New reward axes need a documented mode order, a deterministic
unit test, and an explicit statement of measurement limitations.

Do not commit model weights, checkpoints, generated image batches, access
tokens, W&B credentials, or machine-specific absolute paths.
