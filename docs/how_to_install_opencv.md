# Install OpenCV 

## Install OpenCV-Python under Ubuntu 

Follow the instructions on this [page](https://vitux.com/opencv_ubuntu/).

In order to use OpenCV, run the following command 
```
$ sudo apt install libopencv-dev python3-opencv
$ pip3 install opencv-contrib-python
```


## Install OpenCV-Python under Windows

Follow the instructions on this [page](
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#install-opencv-python-in-windows). 


## Install OpenCV under Anaconda 

You can create a new separate conda environment
```
$ conda create -yn opencvenv python=3.6
$ conda activate opencvenv
$ conda install -c menpo opencv3
```

To deactivate the active environment, use
```
$ conda deactivate
```
This will bring you back to your default conda environment.

### Check OpenCV version once installed

How to check the installed opencv version 
```
$ python -c "import cv2; print(cv2.__version__)"
```