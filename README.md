# PyDVB

PyDVB is a lightweight Python workspace for experimenting with Digital Video
Broadcasting (DVB) signal processing ideas. The goal is to provide a clean
starting point for parsing transport streams, prototyping demodulation
algorithms, and documenting findings as the project grows.

## What’s here

- Minimal Git repository ready for Python packaging
- Room to add notebooks, signal-processing utilities, and CLIs
- Suggested workflow pinned below so new contributors know where to start

## Getting started

1. Ensure you have Python 3.11+ available on your system.
2. Create an isolated environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install project dependencies once they exist:
   ```bash
   pip install -r requirements.txt
   ```

## Development workflow

- Keep Python dependencies tracked in `requirements.txt` (or `pyproject.toml`
  once tooling is in place).
- Add reproducible scripts under `scripts/` for any DSP or analysis helpers.
- Prefer notebooks under `notebooks/` for exploratory signal work; export key
  findings into markdown docs for long-term reference.

## Contributing

1. Create a feature branch.
2. Add or update tests/notebooks as needed.
3. Run `pytest` (or the relevant test tool) before opening a PR.

## License

Add your chosen license here (MIT, Apache-2.0, etc.) when the project direction
is finalized.

