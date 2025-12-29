# PICASSO

[![License](https://img.shields.io/pypi/l/PICASSO.svg?color=green)](https://github.com/Lucaslab-Berkeley/PICASSO/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/PICASSO.svg?color=green)](https://pypi.org/project/PICASSO)
[![Python Version](https://img.shields.io/pypi/pyversions/PICASSO.svg?color=green)](https://python.org)
[![CI](https://github.com/Lucaslab-Berkeley/PICASSO/actions/workflows/ci.yml/badge.svg)](https://github.com/Lucaslab-Berkeley/PICASSO/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Lucaslab-Berkeley/PICASSO/branch/main/graph/badge.svg)](https://codecov.io/gh/Lucaslab-Berkeley/PICASSO)

Projection Improvement by Ctf Aberration, magnification aniSotropy, and Scattering factor Optimization

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork Lucaslab-Berkeley/PICASSO --clone
# or just
# gh repo clone Lucaslab-Berkeley/PICASSO
cd PICASSO
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
