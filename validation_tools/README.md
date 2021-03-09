# Validation tools
----

This folder contains various codes built on top of __*dstack*__ to analysise the output of the pipelines. These scripts are not integral part of the package, but can be a valuable addition for comparing the output from different pipelines (grid- , image- and viusibility stacking).

While the __*dstac.sdiagnostics*__ module functions are used to generate quick and dirty plots to analyse sources found by __*SoFiA*__ and should jointly used with __*radiopadre*__ notebooks, the functions defined here are meant to used as validation tools for pipelines. When a new pipeline is setup its output (namely the found sources) should be testegd against the (old or canonical) previous pipelines.

__NOTE__ that the current version of this directory is pretty hectic as I have some code lying around used to make plots for slides and also some old code used to just test things.

__TO DO:__ clean up this directory and remove all code uses the validation tools as potntional users don't need that, but the tools only.