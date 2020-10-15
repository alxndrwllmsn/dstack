# dstack pipelines
----

This folder contains stacking pipelines for DINGO using __[Snakemake](https://github.com/snakemake/snakemake)__ to chain _Yandasoft_ with __dstack__ aplications. Each subfolder has a specific pipeline, as modularisation of Snakemake pipelines is not required yet. The pipeline is defined by the respectibe ``Snakefile`` in eah subfolder.

Currently pipelines are running on a local HPC cluster.

Useful material for running snakemake pipelines can be found [here](https://hpc-carpentry.github.io/hpc-python/15-snakemake-python/), [here](https://hackmd.io/@bluegenes/BJPrrj7WB), [here](https://www.sichong.site/2020/02/25/snakemake-and-slurm-how-to-manage-workflow-with-resource-constraint-on-hpc/), [here](https://tinyheero.github.io/2019/08/30/wildcards-in-snakemake.html), [here](https://edwards.sdsu.edu/research/wildcards-in-snakemake/) and in the __[official documentation](https://snakemake.readthedocs.io/en/stable/index.html)__.
