# Fed-BioMed dataset

All commands on this page assume the working directory is somewhere inside `<NIPOPPY_ROOT>`.

## Pipeline installation

Install the Nipoppy-to-Fed-BioMed conversion pipeline:

```bash
nipoppy pipeline install 23753437
```

The pipeline should now be listed when you run:

```bash
nipoppy pipeline list
```

<!-- TODO expected output -->

## Dataset extraction

Combine the FreeSurfer IDPs and harmonized phenotypic data into a Fed-BioMed dataset:

```bash
nipoppy process --pipeline nipoppy2fedbiomed --pipeline-step age
nipoppy process --pipeline nipoppy2fedbiomed --pipeline-step sex
```

For each of the prediction tasks (brain age regression and biological sex classification), this will write a CSV file (the Fed-BioMed dataset) and a JSON file used to register it on the [Fed-BioMed node](fedbiomed_node.md).

<!-- TODO tree -->
