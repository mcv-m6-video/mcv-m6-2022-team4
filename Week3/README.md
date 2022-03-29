Task specific files are seperated in the 2 folders, other ones are used mutually.


<h2> Common Files </h2>

<b> metrics.py: </b> Has the common metrics that are used for most tasks: mAP, IOU etc.

<b> read_files.py: </b> Read dataset and annotations, parse them into suitable formats.

<b> utils.py: </b> General functions that are used throughout the weeks, creating gifs, drawing boxes on frames, generating a random color for each box etc.

<h3> Object Detection </h3>

Detectron2 is used for Object Detection tasks. Evaluation is done with CocoEvaluator.

<b> LossEvalHook.py: </b> Customized loss evaluation for detectron2. Gets the validation losses alongside traning losses during finetuning.

<b> pretrained.py: </b> Runs inference on the whole dataset for the pretrained models. 

<b> finetuned.py: </b> Finetuning with Faster R-CNN. Train-Test split strategy should be chosen.

<b> get_ft_preds.py: </b> Run inference on the finetuned network.

<h3> Object Tracking </h3>

<b> sort.py: </b> SORT for kalman filters. More detailed information can be found <a href="https://github.com/abewley/sort"> here. </a>

<b> tracking.py </b> Maximum Overlap tracker.
