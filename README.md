
This repository implements the conditional language generation model of Wen et al. for the DSTC2 restaurant domain data.

The main.lua script is used to for training/evaluation. Adjust the hyperparameter values at the beginning of the script as necessary.

For training, set run\_flow=True, generate\_test=False and run the script as follows

```
th main.lua <gpu_id> <Run description>
```

where ```<gpu\_id>``` indicates the GPU to use and ```<Run description>``` a description of the experiment. The descriptions are logged into ```results/descriptions``` and a folder is created in the results folder corresponding to the run.

Training prints validation error and writes checkpoints to the ```results/<id>/models``` folder where ```<id>``` is the id of the experiment.

Once training is done, for evaluation set run\_flow=False, generate\_test=True. Also set ```pre_init_mdl_path``` to the name of the checkpoint that needs to be evaluated. Run the main script as described above.

Evaluation results will be logged into the folder corresponding to this evaluation run. 

Frequency of logging validation error and creating checkpoints can be varied using the parameters ```*log_freq```.

Wen, T. H., Gasic, M., Mrksic, N., Su, P. H., Vandyke, D., & Young, S., Semantically conditioned lstm-based natural language generation for spoken dialogue systems. In EMNLP, 2015.
