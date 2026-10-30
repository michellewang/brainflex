# brainflex

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/neurodatascience/brainflex/main.svg)](https://results.pre-commit.ci/latest/github/neurodatascience/brainflex/main)

Brain federated learning experiment.

## Development

This project uses [uv](https://docs.astral.sh/uv/).

### Installation

```bash
uv sync
```

### Git hooks via `prek`

```bash
uv run prek install
uv run prek run -a
```

### Tests

```bash
uv run pytest
```

### Documentation

```bash
uv run make -C docs html
cd docs/_build/html
python -m http.server
```

### Docker images

- `vpn-client`: WireGuard VPN client
- `vpn-server`: WireGuard VPN server, built on top of `vpn-client`

### Releases

CLI releases are tagged `cli/vX.Y.Z`.
Docker image releases are tagged `docker/vX.Y.Z`
