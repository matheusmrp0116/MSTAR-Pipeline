# Reproducible SAR Data Pipeline

## Overview

This repository implements a reproducible data pipeline for the ingestion, standardization, validation, and generation of derived Synthetic Aperture Radar (SAR) data.

The MSTAR dataset, that was produced by SNL (Sandia National Laboratories) and published by SDMS (Sensors Data Management System), is the default dataset used in this pipeline, particularly the Mixed Targets part.

The current source-ingestion implementation is designed and validated for MSTAR data stored in Phoenix format. After ingestion, observations are converted to a canonical SAR representation based on magnitude and phase arrays. Subsequent processing and validation stages operate on this canonical representation and do not assume predefined image dimensions.

The pipeline emphasizes:

- source-data integrity;
- deterministic processing;
- provenance and traceability;
- preservation of original image dimensions;
- canonical representation of SAR observations;
- reproducible stochastic processing;
- explicit metadata generation;
- independent validation of generated artifacts.

The repository does **not** include the original SAR source data or generated dataset artifacts.

---

## 1. Pipeline Architecture

The pipeline is organized into four conceptual layers:

```text
Source SAR Data
      │
      ▼
Source-Specific Ingestion
      │
      ▼
Canonical SAR Representation
      │
      ▼
Derived SAR Data
      │
      ▼
Validation and Metadata
```

For the current implementation:

```text
MSTAR source files
      │
      ▼
Phoenix header parsing
Binary layout validation
Source byte-order validation
      │
      ▼
Canonical SAR representation
magnitude : float32
phase     : float32
original rows × columns
      │
      ▼
Complex reconstruction
      │
      ▼
Speckle simulation
      │
      ▼
Derived Layer
      │
      ▼
Independent validation
```

The source-specific part of the current implementation is associated with MSTAR/Phoenix ingestion. The canonical and derived layers are designed to operate independently of predefined image dimensions.

---

## 2. Repository Structure

```text
mstar-pipeline/
│
├── data/
│   ├── raw/
│   │   └── mstar/
│   │
│   ├── standardized/
│   │
│   ├── derived/
│   │   ├── speckle/
│   │   └── speckle_container_test/
│   │
│   └── metadata/
│
├── src/
│   └── mstar_pipeline/
│       ├── __init__.py
│       ├── config.py
│       ├── hashing.py
│       ├── inventory.py
│       ├── phoenix.py
│       ├── enrich_manifest.py
│       ├── signal.py
│       ├── standardize.py
│       ├── catalog.py
│       ├── complex_signal.py
│       ├── scattering.py
│       └── derived_speckle.py
│
├── tests/
│   └── test_phoenix.py
│
├── run_inventory.py
├── run_header_enrichment.py
├── run_signal_layout_test.py
├── run_layout_collection_diagnostic.py
├── run_endianness_collection_diagnostic.py
├── run_standardization_test.py
├── run_standardization.py
├── run_reproducibility_test.py
├── run_catalog_validation.py
├── run_complex_roundtrip_validation.py
├── run_scattering_model_validation.py
├── run_mstar_speckle_integration_pilot.py
├── run_speckle_convergence_validation.py
├── run_derived_speckle_container_test.py
├── run_derived_speckle_generation.py
├── run_derived_layer_validation.py
│
├── environment-v1.0.txt
└── requirements-v1.0.txt
```

Generated files under `data/standardized/`, `data/derived/`, and `data/metadata/` should not be treated as manually maintained source files.

---

## 3. Input Data

### 3.1 Source data location

Place the original MSTAR data under:

```text
data/raw/
```

The source directory structure may contain nested folders. The pipeline recursively discovers source files where applicable.

Example:

```text
data/
└── raw/
    └── <original MSTAR directory structure>
```

The exact directory names below `data/raw/` are not intended to encode scientific assumptions about the dataset.

### 3.2 Preserve the original source data

Source files should be preserved exactly as distributed.

Do not:

- rename source observations;
- convert the binary data before ingestion;
- resize images;
- crop or pad observations;
- manually change byte order;
- manually extract magnitude or phase;
- modify Phoenix headers;
- overwrite original source files;
- remove auxiliary source files solely because they are not SAR observations.

The Raw Layer is treated as immutable.

Cryptographic hashes are generated to support source integrity and provenance verification.

---

## 4. Current Input Format Support

The current ingestion layer is implemented and validated for MSTAR data using Phoenix headers and the corresponding binary signal organization.

The current implementation includes:

- Phoenix header parsing;
- header-length inspection;
- binary-layout validation;
- source signal-size validation;
- source byte-order diagnostics;
- magnitude and phase plane extraction;
- conversion to the canonical SAR representation.

Support for a different SAR source format may require a source-specific reader or adapter.

The downstream canonical and derived layers should not be interpreted as proof that every SAR product can be ingested without adaptation.

---

## 5. Canonical SAR Representation

Each successfully standardized SAR observation is represented using:

```text
magnitude : float32 array
phase     : float32 array
```

The original spatial dimensions are preserved.

No fixed image size is required by the canonical representation.

For example, the pipeline may process observations with different combinations of:

```text
rows × columns
```

provided that the source-ingestion layer can correctly interpret them.

The standardization stage does not perform model-specific normalization, resizing, padding, or cropping.

The canonical representation is intended to separate source-specific binary interpretation from subsequent processing.

---

## 6. Environment

The reference implementation was executed using Python in an Anaconda environment.

Environment information can be recorded using files such as:

```text
environment-v1.0.txt
requirements-v1.0.txt
```

The package inventory can be generated with:

```bash
python -m pip list --format=freeze > requirements-v1.0.txt
```

For Conda environments, additional environment records may be generated with:

```bash
conda list > conda-packages-v1.0.txt
```

and:

```bash
conda env export > environment-conda-v1.0.yml
```

These files provide provenance information about the software environment used for dataset generation and validation.

---

## 7. Running the Pipeline

Run all commands from the repository root.

Example:

```bash
cd mstar-pipeline
```

The stages should be executed in the order described below.

---

## 8. Stage 1 — Raw Data Inventory

Run:

```bash
python run_inventory.py
```

This stage recursively inventories the Raw Layer and records source provenance information.

Typical metadata include:

- source identifier;
- original filename;
- relative source path;
- file size;
- SHA-256 source hash;
- pipeline version;
- inventory timestamp.

The resulting inventory is written to the metadata layer.

No source file is modified.

---

## 9. Stage 2 — Source Metadata Enrichment

Run:

```bash
python run_header_enrichment.py
```

This stage inspects candidate source observations and enriches the inventory with Phoenix metadata.

Files that are not SAR observations may be classified separately as auxiliary source objects.

The enrichment process distinguishes between:

```text
observation
auxiliary
unresolved
```

where applicable.

This prevents non-observation files distributed with the source dataset from being incorrectly interpreted as parsing failures.

---

## 10. Stage 3 — Binary Layout Test

Run:

```bash
python run_signal_layout_test.py
```

This stage performs a deterministic test of the source binary organization using a valid observation discovered from the available Raw Layer.

The routine does not require a predefined image dimension.

The test inspects properties such as:

- Phoenix header offset;
- rows and columns;
- binary payload size;
- signal-plane organization;
- candidate byte-order interpretations.

This is a diagnostic step and should be completed before collection-wide binary processing.

---

## 11. Stage 4 — Collection-Wide Layout Validation

Run:

```bash
python run_layout_collection_diagnostic.py
```

This stage evaluates the binary-layout assumptions across the complete set of recognized SAR observations.

Its purpose is to verify that the source organization expected by the ingestion implementation is consistently satisfied.

---

## 12. Stage 5 — Source Byte-Order Validation

Run:

```bash
python run_endianness_collection_diagnostic.py
```

This stage evaluates candidate byte-order interpretations across the source observations.

The resulting diagnostics provide empirical evidence for the byte order used by the current source reader.

Byte order is a property of the source representation and should not be inferred from the host computer architecture.

---

## 13. Stage 6 — Standardization Test

Run:

```bash
python run_standardization_test.py
```

This stage tests the conversion of a source observation into the canonical representation.

The canonical artifact contains:

```text
magnitude
phase
```

stored as `float32` arrays while preserving the original observation dimensions.

---

## 14. Stage 7 — Standardized Layer Generation

Run:

```bash
python run_standardization.py
```

This stage generates the canonical Standardized Layer for all valid SAR observations.

The standardized artifacts are stored under:

```text
data/standardized/
```

The standardization process preserves:

- source identity;
- original spatial dimensions;
- magnitude information;
- phase information;
- provenance relationships;
- deterministic artifact generation.

No resizing, padding, cropping, or model-specific normalization is applied.

---

## 15. Stage 8 — Reproducibility Validation

Run:

```bash
python run_reproducibility_test.py
```

This stage regenerates standardized representations from the Raw Layer and compares them with the existing standardized artifacts.

The validation evaluates properties such as:

- source hash consistency;
- magnitude equality;
- phase equality;
- standardized artifact reproducibility.

This provides evidence that the standardized dataset can be reproduced from the original source data using the same pipeline.

---

## 16. Stage 9 — Dataset Catalog Validation

Run:

```bash
python run_catalog_validation.py
```

This stage creates and validates the observation-level dataset catalog.

The catalog connects:

```text
source observation
        │
        ▼
source metadata
        │
        ▼
standardized observation
        │
        ▼
reproducibility evidence
```

The catalog provides a central provenance index for subsequent pipeline stages.

---

## 17. Stage 10 — Complex Representation Validation

Run:

```bash
python run_complex_roundtrip_validation.py
```

The canonical magnitude and phase arrays are used to construct a complex representation.

Conceptually:

```text
magnitude + phase
        │
        ▼
complex representation
        │
        ▼
magnitude + phase
```

The round-trip validation quantifies numerical differences introduced by finite-precision arithmetic.

Phase comparisons account for circular phase equivalence and for pixels where phase is undefined because magnitude is zero.

---

## 18. Stage 11 — Stochastic Speckle Model Validation

Run:

```bash
python run_scattering_model_validation.py
```

This stage validates the statistical implementation of the coherent equivalent-scatterer speckle generator independently of the SAR observations.

The implemented experimental conditions are:

```text
K_N01
K_N02
K_N05
FDS
```

where the finite-scatterer conditions represent different numbers of coherent equivalent scatterers and `FDS` provides a fully developed speckle reference condition.

The validation examines statistical behavior such as:

- mean intensity;
- intensity variance;
- coefficient of variation;
- normalized intensity moments;
- agreement with theoretical reference moments.

This stage validates the implementation of the stochastic generator. It should not be interpreted by itself as validation of the physical suitability of a particular model condition for every SAR dataset.

---

## 19. Stage 12 — SAR/Speckle Integration Pilot

Run:

```bash
python run_mstar_speckle_integration_pilot.py
```

This stage evaluates the integration between the canonical SAR representation and the stochastic speckle generator.

The routine discovers the image dimensions available in the dataset catalog and selects observations deterministically.

It does not require predefined dimensions such as 54×54 or 128×128.

The pilot evaluates the transformation conceptually represented by:

```text
baseline complex SAR
        ×
stochastic speckle field
        │
        ▼
derived complex SAR
```

while preserving provenance and deterministic seed generation.

---

## 20. Stage 13 — Monte Carlo Convergence Validation

Run:

```bash
python run_speckle_convergence_validation.py
```

This stage evaluates convergence of the stochastic simulation using increasing numbers of realizations.

The routine automatically discovers the distinct image dimensions available in the current dataset and deterministically selects a representative observation for each dimension.

No particular spatial dimension is required.

The same initial sequence of stochastic realizations is reused across convergence checkpoints, allowing nested Monte Carlo comparisons.

The analysis evaluates pixel-level and global-power convergence behavior and estimates empirical convergence rates.

---

## 21. Stage 14 — Derived Container Test

Run:

```bash
python run_derived_speckle_container_test.py
```

Before generating the complete Derived Layer, this stage tests the physical container architecture.

The routine:

1. discovers the image dimensions represented in the current dataset;
2. selects one parent observation per available dimension;
3. generates all configured stochastic conditions;
4. stores multiple logical realizations in each physical container;
5. reloads the containers;
6. validates their internal arrays and identifiers;
7. checks deterministic reproduction.

Test artifacts are written to a directory separate from the final Derived Layer:

```text
data/derived/speckle_container_test/
```

This prevents test artifacts from being mixed with final generated data.

---

## 22. Stage 15 — Derived Layer Generation

Run:

```bash
python run_derived_speckle_generation.py
```

This stage generates the complete Derived Layer.

The output is organized using physical containers that store multiple stochastic realizations for each parent observation and experimental condition.

A container conceptually stores:

```text
magnitude
    shape: (realizations, rows, columns)
    dtype: float32

phase
    shape: (realizations, rows, columns)
    dtype: float32

seeds
    one deterministic seed per realization

realization_ids
    one logical identifier per realization
```

The number of generated physical containers and logical derived observations depends on:

- the number of valid parent observations;
- the configured stochastic conditions;
- the configured number of realizations.

These values are derived from the actual dataset and pipeline configuration rather than from predefined dataset cardinalities.

---

## 23. Deterministic Stochastic Processing

Stochastic generation uses deterministic seeds derived from provenance information.

The seed-generation policy incorporates identifiers such as:

```text
parent source
condition
realization
model version
pipeline version
```

This allows the same logical derived observation to be regenerated deterministically under the same pipeline configuration.

The stochastic seed is therefore part of dataset provenance.

---

## 24. Stage 16 — Independent Derived Layer Validation

Run:

```bash
python run_derived_layer_validation.py
```

The final validation is divided into four blocks.

### D1 — Cardinality and Structure

D1 validates structural relationships derived dynamically from the current dataset.

It checks properties such as:

- parent coverage;
- number of conditions per parent;
- logical realizations per container;
- uniqueness of container identifiers;
- uniqueness of logical derived identifiers;
- logical coordinate uniqueness;
- logical-to-physical container relationships;
- array indices;
- realization identifiers;
- dimension distribution.

Expected cardinalities are calculated from the current dataset catalog and pipeline configuration.

They are **not hard-coded for a specific dataset**.

### D2 — Physical Integrity

D2 verifies the physical derived artifacts.

It checks:

- container existence;
- SHA-256 integrity;
- correspondence between stored and recorded hashes.

### D3 — Container Content

D3 independently opens the generated containers and validates:

- required arrays;
- array shapes;
- `float32` canonical data types;
- finite magnitude values;
- finite phase values;
- realization identifiers;
- deterministic seeds;
- correspondence with the logical manifest.

Expected spatial shapes are obtained from the metadata of each individual observation.

No predefined image dimensions are assumed.

### D4 — Independent Deterministic Reproduction

D4 independently regenerates selected derived observations.

The routine:

1. discovers all image dimensions represented in the current dataset;
2. deterministically selects up to a configured number of parent observations from each available dimension;
3. selects configured realization indices;
4. reconstructs the parent complex representation;
5. regenerates the stochastic field from the deterministic seed;
6. recreates the derived observation;
7. compares the regenerated magnitude and phase with the stored arrays.

The number of D4 tests therefore adapts to the dimensions and observations available in the current dataset.

---

## 25. Validation Framework

The pipeline distinguishes three different validation levels.

### A. Source and Canonical Dataset Validation

The source-to-canonical pipeline evaluates:

```text
V1 — Integrity
V2 — Structural consistency
V3 — Provenance
V4 — Reproduction
V5 — Catalog and metadata consistency
```

These validations concern the reproducibility and integrity of the source and standardized dataset.

### B. Stochastic Model Validation

The stochastic generator is evaluated separately from the SAR observations.

This distinction is important because software/statistical validation of the generator is not equivalent to demonstrating that a particular stochastic condition is physically appropriate for every SAR product.

### C. Derived Dataset Validation

The generated Derived Layer is evaluated using:

```text
D1 — Cardinality and structure
D2 — Physical integrity
D3 — Container content
D4 — Independent deterministic reproduction
```

This separation provides explicit evidence for different components of dataset reproducibility.

---

## 26. Data Provenance

The intended provenance chain is:

```text
Raw source object
        │
        ├── SHA-256
        ├── source identifier
        └── source metadata
        │
        ▼
Canonical standardized observation
        │
        ├── standardized identifier
        ├── canonical arrays
        ├── dimensions
        ├── artifact hash
        └── pipeline version
        │
        ▼
Derived container
        │
        ├── parent source identifier
        ├── condition
        ├── model version
        ├── pipeline version
        ├── deterministic seeds
        ├── container hash
        └── realization mapping
        │
        ▼
Logical derived observations
```

The objective is to make each generated observation traceable to its source and processing configuration.

---

## 27. Generated Metadata

Depending on the execution stage, the `data/metadata/` directory contains artifacts such as:

```text
manifest.parquet
manifest_enriched.parquet
endianness_diagnostic.parquet
standardization_manifest.parquet
reproducibility_validation.parquet
dataset_catalog.parquet

speckle_convergence_validation.parquet
speckle_convergence_slopes.parquet

derived_container_test_manifest.parquet
derived_container_test_realizations.parquet

derived_speckle_containers.parquet
derived_speckle_manifest.parquet

derived_validation_d1_structure.parquet
derived_validation_d2_integrity.parquet
derived_validation_d3_content.parquet
derived_validation_d4_reproducibility.parquet
derived_layer_validation_summary.parquet
```

Additional diagnostic or error files may be generated when applicable.

---

## 28. Recommended First-Time Execution Order

For a complete first execution, use:

```bash
python run_inventory.py
python run_header_enrichment.py
python run_signal_layout_test.py
python run_layout_collection_diagnostic.py
python run_endianness_collection_diagnostic.py
python run_standardization_test.py
python run_standardization.py
python run_reproducibility_test.py
python run_catalog_validation.py
python run_complex_roundtrip_validation.py
python run_scattering_model_validation.py
python run_mstar_speckle_integration_pilot.py
python run_speckle_convergence_validation.py
python run_derived_speckle_container_test.py
python run_derived_speckle_generation.py
python run_derived_layer_validation.py
```

Do not proceed to complete Derived Layer generation if earlier structural, standardization, reproducibility, stochastic-model, or container tests reveal unresolved errors.

---

## 29. Clean First-Time Repository State

Before the first execution, the data directories may be empty:

```text
data/
├── raw/
├── standardized/
├── derived/
└── metadata/
```

After obtaining the original source dataset, place it under:

```text
data/raw/
```

The other directories are populated by the pipeline.

If empty directories need to be preserved in version control, placeholder files such as `.gitkeep` may be used.

---

## 30. Re-running the Pipeline

The Raw Layer should remain immutable.

Before regenerating a dataset from scratch, generated artifacts from a previous execution should not be silently mixed with a new input dataset.

In particular, ensure that the contents of:

```text
data/standardized/
data/derived/
data/metadata/
```

correspond to the same execution lineage and source dataset.

For a clean regeneration, archive or remove previous generated artifacts before starting again, while preserving the original Raw Layer.

---

## 31. Dataset Independence and Current Scope

The pipeline no longer assumes specific SAR image dimensions in the downstream processing and validation stages.

Dataset cardinalities and dimension distributions are discovered dynamically from the current input and generated metadata.

However, this distinction is important:

```text
dimension-independent
        ≠
source-format-independent
```

The current source-ingestion implementation remains designed for MSTAR/Phoenix data.

Consequently, a different SAR dataset may require a source-specific ingestion adapter to convert its native representation into the canonical representation expected by the downstream pipeline.

Once a compatible canonical representation is available, the subsequent architecture is designed to avoid assumptions about specific image dimensions or fixed dataset cardinalities.

---

## 32. Extending the Pipeline to Other SAR Sources

A future extension can introduce source adapters with a common conceptual interface:

```text
SAR source
    │
    ▼
source-specific reader
    │
    ▼
canonical magnitude + phase representation
    │
    ▼
common downstream pipeline
```

For example:

```text
MSTAR/Phoenix ──────┐
                    │
Other SAR format ───┼──► Canonical SAR Layer
                    │
Another source ─────┘
                            │
                            ▼
                    common processing
                            │
                            ▼
                     Derived Layer
```

A new source adapter would be responsible for correctly interpreting the native data format, metadata, byte organization, signal representation, and other source-specific properties.

Such support should be independently validated before claiming compatibility with a new SAR dataset.

---

## 33. Reproducibility Principles

The pipeline follows the following principles:

1. **Immutable Raw Layer**  
   Original source data are not modified.

2. **Cryptographic provenance**  
   Source and generated artifacts are associated with SHA-256 hashes where applicable.

3. **Deterministic identifiers**  
   Dataset objects are assigned reproducible identifiers.

4. **Canonical representation**  
   Source-specific signal representation is converted to a common magnitude/phase representation.

5. **Dimension preservation**  
   Original spatial dimensions are preserved rather than forced into a predefined shape.

6. **Deterministic stochastic processing**  
   Random generation is controlled through provenance-derived seeds.

7. **Explicit metadata**  
   Relationships among source, standardized, and derived observations are recorded.

8. **Independent validation**  
   Generated artifacts are reopened, checked, and independently reproduced where applicable.

9. **Separation of validation levels**  
   Source integrity, stochastic-model validation, and Derived Layer validation are treated as distinct methodological components.

10. **No dataset-specific cardinality assumptions**  
    Expected counts are derived from the current input dataset and pipeline configuration rather than hard-coded from a reference execution.

---

## 34. Important Interpretation Note

Successful execution of the pipeline demonstrates computational consistency and reproducibility under the implemented data model and configuration.

It does not, by itself, establish that:

- every SAR source format is directly compatible with the current ingestion layer;
- every SAR product provides equivalent magnitude and phase information;
- every stochastic speckle condition is physically appropriate for every SAR dataset;
- validation on one source dataset automatically establishes validity for another source.

Compatibility with additional SAR sources should therefore distinguish:

```text
software portability
        from
scientific validity
```

Both should be evaluated when extending the pipeline.

---

## 35. Citation and Dataset Release

When releasing a dataset generated with this pipeline, the release should document at minimum:

- source dataset and access information;
- pipeline version;
- source-ingestion assumptions;
- software environment;
- stochastic-model version;
- configuration used for generation;
- validation outputs;
- provenance metadata;
- repository release or commit identifier.

A permanent repository release should be associated with the dataset publication whenever possible.

---

## 36. License

Code: MIT License
Data: CC-BY-4.0

The software license does not replace or modify the license, access conditions, or redistribution restrictions of the original SAR source dataset.

Users are responsible for complying with the terms applicable to the source data.

---

## 37. Contact

matheus.pessanha.102678@ga.ita.br
satolo@ita.br
carmen.beld@gmail.com