# CS172-Final-Project
Full Name(姓名): 毕春浩 杨宏笛 吴沁蔓 李嘉轩

Student ID(学号): 2019533135 2019533234 2019533118 2019533148

Because the model is large, maybe git Large File Storage is needed.
After cloning this repository, you may try demo to get a video pose estimation result.
Put your video in the current directory and change YOUR_VIDEO_NAME to your .mp4 video name.

(Maybe some preliminary python packages are required)
#### To run demo
```
python tools/demo.py --cfg experiments/mine_data/final.yaml \
    --videoFile ./YOUR_VIDEO_NAME.mp4 \
    --outputDir output \
    TEST.MODEL_FILE model/model_best_new.pth.tar
```

The above command will create a video under *output* directory and a lot of pose image under *output/pose* directory. 

