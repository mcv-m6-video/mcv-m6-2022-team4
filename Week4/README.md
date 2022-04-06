Tasks are done in different conda environments, each one has its own yml file.

<h4> Task 1: Optical Flow </h4>

<h4> Task 2: MTSC </h4>

Network training and inference are done with <a href="https://detectron2.readthedocs.io/en/latest/index.html"> Detectron2 </a>.

<h5> Pipeline: </h5>

<ol>
  <li> Use finetuned Faster R-CNN from last week to create labels for each frame. (create_boxes.py)
  <li> With created labels, train the network for each sequence pair. (One sequence is chosen as test, other two are used for training) (finetune.py)
  <li> Get the trained detections for each camera in the test sequence. (get_cam_preds.py)
  <li> Use Kalman to get tracking results and rvaluate them with py-motmetrics. (get_mtsc_results.py) 
</ol>
