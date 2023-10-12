# Video Human Pose Estimation based on DEKR
Team: Chunhao Bi Hongdi Yang Qinman Wu Jiaxuan Li

## This is the course project of CS171 Computer Vision in ShanghaiTech University. Further details about this project can be found at our [report](./report.pdf).



Because the model is large, maybe git Large File Storage is needed.
After cloning this repository, you may try demo to get a video pose estimation result.
Put your video in the current directory and change YOUR_VIDEO_NAME to your .mp4 video name.

(Maybe some preliminary python packages are required)
#### To run demo (Currently not available)
```
python tools/demo.py --cfg experiments/mine_data/final.yaml \
    --videoFile ./YOUR_VIDEO_NAME.mp4 \
    --outputDir output \
    TEST.MODEL_FILE model/model_best_new.pth.tar
```

The above command will create a video under *output* directory and a lot of pose image under *output/pose* directory. 
```
