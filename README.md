
This repository implements the conditional language generation model of Wen et al. for the DSTC2 restaurant domain data.

Data: Download DSTC2 data from ```http://camdial.org/~mh521/dstc/```. Create a folder named ```data``` in the same level as the src folder. Organize the downloaded data so that the data folder will have sub-directories named config, train and test where the train and test directories will consist of the actual data ("Mar*") from the relevant zip files.

The main.lua script is used to for training/evaluation. Adjust the hyperparameter values at the beginning of the script as necessary.

For training, set run\_flow=True, generate\_test=False and run the script as follows

```
th main.lua <gpu id> <Run description>
```

where ```<gpu id>``` indicates the GPU to use and ```<Run description>``` a description of the experiment. The descriptions are logged into ```results/descriptions```, an id number is assigned to the current run, and a folder is created in the results folder corresponding to the id.

Training prints validation error and writes checkpoints to the ```results/<id>/models``` folder where ```<id>``` is the id of the experiment.

Once training is done, for evaluation set run\_flow=False, generate\_test=True. Also set ```pre_init_mdl_path``` to the name of the checkpoint that needs to be evaluated. Run the main script as described above.

Evaluation results will be logged into the folder corresponding to this evaluation run. 

Frequency of logging validation error and creating checkpoints can be varied using the parameters ```*log_freq```.

Wen, T. H., Gasic, M., Mrksic, N., Su, P. H., Vandyke, D., & Young, S., Semantically conditioned lstm-based natural language generation for spoken dialogue systems. In EMNLP, 2015.
