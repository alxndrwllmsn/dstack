# Tutorial to building grid stacking pipelines
----
<p align="center">
_**by using Snakemake and dstack**_
</p>
----
This is a crash-course to build grid stacking pipelines using __[dstack](https://github.com/rstofi/dstack)__ __[YandaSoft](https://github.com/ATNF/yandasoft)__ and __[Snakemake](https://github.com/snakemake/snakemake)__. The proper way, of course, would be to read the documentation for all of the software mentioned above before getting started. Ergo this material will provide only the technical details on how to build a specific pipeline.

Useful material for running snakemake pipelines can be found [here](https://hpc-carpentry.github.io/hpc-python/15-snakemake-python/), [here](https://hackmd.io/@bluegenes/BJPrrj7WB), [here](https://www.sichong.site/2020/02/25/snakemake-and-slurm-how-to-manage-workflow-with-resource-constraint-on-hpc/), [here](https://tinyheero.github.io/2019/08/30/wildcards-in-snakemake.html), [here](https://edwards.sdsu.edu/research/wildcards-in-snakemake/) and in the __[official documentation](https://snakemake.readthedocs.io/en/stable/index.html)__.

Each sub-folder builds the same example pipeline, but with different complexity starting from a skeleton pipeline doing nothing and can be run on a local machine to set up an actual working grid stacking pipeline running on an HPC environment. Currently, the following tutorial steps are implemented:

- simple_snakemake_pipeline: a simple pipeline showing the syntax and how to run pipelines