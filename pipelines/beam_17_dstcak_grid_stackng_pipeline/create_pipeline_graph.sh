#/bin/bash

#Simple pipeline graph:
#snakemake --rulegraph | dot -Tpng > pipeline_graph.png

#Complex_graph
snakemake --dag | dot -Tpng > pipeline_graph.png
