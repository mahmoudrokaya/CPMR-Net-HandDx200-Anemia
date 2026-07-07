# CPMR-Net HandDx-200 Anemia

This repository contains the publication-ready implementation of the project:

**Cooperative Physiological Multi-Representation Network (CPMR-Net) for participant-level multimodal anemia diagnosis from hand images.**

## Repository Scope

This GitHub repository intentionally contains only the lightweight reproducibility package:

- `Codes/`
- `README.md`
- `LICENSE`
- `requirements.txt`
- `configs/`
- `metadata/`
- `example_results/`

Large generated files, raw image datasets, PDFs, archives, trained checkpoints, and image outputs are excluded.

## Key Principle

All experiments are participant-level. Images and derived representations from the same participant are never split across training, validation, test, or cross-validation folds.

## Main Components

- Dataset audit and verification
- Thermal decoding
- Physiological representation generation
- Handcrafted feature extraction
- Statistical validation
- Nonredundant feature selection
- Classical ML benchmarks
- CPMR-Net architecture
- Contrastive pretraining
- Progressive fine-tuning
- Holdout and repeated participant-level cross-validation
- Scientific decision audit

## Data

The HandDx-200 dataset must be obtained from its original public source. This repository does not redistribute the full image dataset.

## License

MIT License.
