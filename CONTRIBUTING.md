# Contributing to PredictMix

Thank you for considering contributing to **PredictMix**!

This project aims to provide a robust, research-grade pipeline for integrated
polygenic and clinical risk prediction. Contributions that improve robustness,
usability, documentation, or scientific relevance are very welcome.

---

## Ways to Contribute

- 🐛 Report bugs
- 📚 Improve documentation and examples
- 🧪 Add tests or improve testing infrastructure
- 🧠 Add new models, feature-selection methods or plotting utilities
- 🔬 Add domain-specific examples (e.g. sickle-cell disease, cardiometabolic traits)

---

## Reporting Issues

If you find a bug or have a feature request:

1. Check if an issue already exists.
2. If not, open a new issue and include:
   - A clear description of the problem or feature.
   - Steps to reproduce (for bugs).
   - Minimal code snippet or dataset description if relevant.
   - Your Python version and `predictmix` version.

---

## Development Setup

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/predictmix.git
   cd predictmix
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in editable/development mode:

   ```bash
   pip install --upgrade pip
   pip install -e .[dev]
   ```

   (If `[dev]` is not defined yet, simply use `pip install -e .`.)

---

## Code Style

- Follow **PEP8** for Python code style.
- Use **type hints** where possible.
- Prefer small, well-documented functions and classes.

---

## Testing

If tests are present (e.g. in `tests/`), run them with:

```bash
pytest
```

Please ensure that new features or bug fixes include appropriate tests whenever possible.

---

## Submitting a Pull Request

1. Create a feature branch:

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes and commit with clear messages.

3. Push to your fork:

   ```bash
   git push origin feature/my-new-feature
   ```

4. Open a Pull Request from your branch to the `main` branch of the main repository.

Please describe:

- What you changed
- Why (motivation)
- How it was tested

---

## Contact

For scientific or design questions:

- **Etienne Ntumba Kabongo** – etienne.kabongo@mcgill.ca  
- **Prof. Emile R. Chimusa** – emile.chimusa@northumbria.ac.uk  

We appreciate your interest in improving PredictMix!