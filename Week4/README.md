Tasks are done in different conda environments, each one has its own yml file.

<h4> Task 1: Optical Flow </h4>

Both Task 1.1 and 1.2 are done in the of_bm_and_ots.py file. Code runs grid search for block matching algorithm and writes the results in a csv file. MSEN, PPEN, and inference time is calculated for each parameter combination. It then runs off-the-shelf methods and shows the quiver plot of optical flow.

<h5> Block Matching Search Grid </h5>
    block_size = 4, 8, 16, 32, 64  <br>
    search_area = 8, 16, 32, 64, 128 <br>
    comp = "forward", "backward" <br>
    loss = 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'

<h5> Tried Off-the-Shelf Methods </h5>
<ul>
  <li> <a href="https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html"> Lucas-Kanade </a>
  <li> <a href="https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af"> Farneback </a>
  <li> <a href="https://github.com/pathak22/pyflow"> PyFlow </a>
</ul>

<h4> Task 2: MTSC </h4>

Network training and inference are done with <a href="https://detectron2.readthedocs.io/en/latest/index.html"> Detectron2 </a>.

<h5> Pipeline: </h5>

<ol>
  <li> Use finetuned Faster R-CNN from last week to create labels for each frame. (create_boxes.py)
  <li> With created labels, train the network for each sequence pair. (One sequence is chosen as test, other two are used for training) (finetune.py)
  <li> Get the trained detections for each camera in the test sequence. (get_cam_preds.py)
  <li> Use Kalman to get tracking results and rvaluate them with py-motmetrics. (get_mtsc_results.py) 
</ol>
