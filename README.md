# Repository for Final Assessment ITB 2019
## Topic: Video Interpolation

## How To Use
1. Download all required library
2. Download pretrained models:
    - Sepconv from [here](http://content.sniklaus.com/sepconv/network-lf.pytorch) then put the downloaded file inside `./SepConv` folder 
    - Deoldify from [here](https://www.dropbox.com/s/336vn9y4qwyg9yz/ColorizeVideo_gen.pth?dl=0) then put the downloaded file in `./DeOldify/models` (Create the folder if necessary)
3. Place your input video in `./video/source/input.mp4`
4. run `python main.py --colorize` if your input is black and white, and `python main.py` if you only want to improve FPS.

## references
```
[1]  @inproceedings{Niklaus_ICCV_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Separable Convolution},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2017}
     }
```