## Instructions to run the code

* Check the versions of the packages in last cell of notebooks printed with `%watermark --iversions`.

## Common issues
* If latex error is raised during plotting the images, try to disable latex by setting `use_latex=False` in `latexify` function.

## Other files

* `experiment.py` to run experiment for various methods (heinonen, delta_inducing)
* `ablations.py` to run ablation experiments with 8 x 8 combination of datasets and methods
* In future, to investigate the behavior of the methods, see `post_run.ipynb`