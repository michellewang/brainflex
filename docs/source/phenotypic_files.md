# Phenotypic files

## Annotation

Download the following files to your local machine:

- `nipoppy_study/tabular/pheno.tsv`
- `nipoppy_study/tabular/pheno.json` (if present)

Then, go through the [Neurobagel annotation tool](https://annotate.neurobagel.org/) workflow using these files.
The following columns need to be annotated:

- Participant ID
- Session ID
- Age
- Sex

Save the annotated data dictionary to `nipoppy_study/tabular/pheno_annotated.json`.

## Harmonization

Harmonize the phenotypic data using Neurobagel's CLI:

```bash
bagel harmonize-pheno \
    --pheno nipoppy_study/tabular/pheno.tsv \
    --dictionary nipoppy_study/tabular/pheno_annotated.json \
    --output nipoppy_study/tabular/harmonized.tsv
```
