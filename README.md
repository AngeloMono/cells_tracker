# Cells tracker

Cells tarckers is a Python project to calculate the cilia beat of cilied cells.

## Installation
This project requires **Python 3.8** version, it does not ensure correct works with later versions.

Step installations:
- Download the project on yours device
    ```bash
    git clone --recurse-submodules https://github.com/AngeloMono/cells_tracker.git
    ```
- Make as a python root the directories: `yolov5`, `yolact` and `sort`
- Install the latest `Cuda` version if your device has a GPU
- Install the correct `torch=1.7` from the [official site](https://pytorch.org/)
- Open a command prompt, go into the root project directory and run the following command to install the requirements
 dependencies saved in [requirements.txt](requirements.txt) 
    ```bash
    pip install -r requirements.txt
    ```
- Download the [weights](https://drive.google.com/file/d/1XmJ-aco5xJdpxPwDBCYYYwwaPWOlcmnH/view?usp=sharing),
 extract the files and paste them in the `~/cells_tracker/weights` folder

## Usage

Open a command prompt and run the command `python main.py` followed by the parameter `--source` (or `-s`) and 
the path of the video source to analyze.

```bash
python main.py --source  'The path file'  # video
                          0               # webcam
```

The executions of the previous command save in the `~/cells_tracker/results` a folder with the name of the analyzed file 
containing a `video.mp4` and , for each cell matched in the video source, 
a plot representing the movement made by the cell's cilia during the video 
(where no movement has been calculated, the points inside the plot are showing in red).
<p align="center">
    <a href="https://drive.google.com/uc?export=view&id=1GtDSqJdUMtxpQE4YwvEU9mWkOcR7B_HR">
        <img src="https://drive.google.com/uc?export=view&id=1GtDSqJdUMtxpQE4YwvEU9mWkOcR7B_HR" style="width: 650px; max-width: 100%; height: auto" title="Movement plot" />
    </a>
</p>

Run the `--help` ( or `-h`) command to see everything you can do.
```bash
python main.py --help 
```

## Debug mode
Adding the parameter `--debug` it is possible to activate the debug mode which shows the processing of the various 
frames step by step.
```bash
python main.py --source 'The path file' --debug
```


In particular it shows:
    <ul>
        <li>a window with the yolo detections and their threshold</li>
        <li>a window with all the tracks managed by the sort class</li>
        <li>a window with the matched tracks by the given yolo detections and the tracks reputed reliable</li>
        <li>for each id in `--ids` a a window with the mask get by yolact, the points taken into consideration for the movement
            and the projection of movement on the perpendicular axis to the junction of the cilia</li>
    </ul>



In debug mode is possible play or stop the video elaboration by the press of the space bar or 
go to the next frame by pressing any key

## Create custom weights
* To create custom weights for Yolov5 visit the
 [YOLOv5_custom_weights.ipynb](https://github.com/AngeloMono/YOLOV5-custom-weights.git) file.
* To create custom weights for Yolact visit the 
 [YOLACT_custom_weights.ipynb](https://github.com/AngeloMono/YOLACT-custom-weights.git) file.


## Contributing
The whole project was developed by Angelo Monopoli.

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Citations
This project use the following github project:
* [Yolov5](https://github.com/ultralytics/yolov5):
    citation page  at the following [link](https://zenodo.org/record/4418161)
* [SORT](https://github.com/abewley/sort)
    ```bash
    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }
    ```
* [Yolact](https://github.com/dbolya/yolact)
    ```bash
    @inproceedings{yolact-iccv2019,
      author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
      title     = {YOLACT: {Real-time} Instance Segmentation},
      booktitle = {ICCV},
      year      = {2019},
    }
    ```
