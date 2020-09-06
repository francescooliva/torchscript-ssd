# TorchScript on NVIDIA GPU

This demo application that shows how to run a network using LibTorch C++ API, is a slightly modificated version
of the original [torchscript-ssd](https://github.com/tufei/torchscript-ssd) project.

This project is supposed to be used together with
[pytorch-ssd](https://github.com/tufei/pytorch-ssd), which can dump the TorchScript.

## Prerequisites

* [LibTorch](pytorch.org) PyTorch C++ API, Cmake, OpenCV

## Build
You can use and IDE with Cmake support enabled like Kdev, VS Code or MS Visual Studio, or from command line:

```bash
mkdir build && cd build
cmake..
```
You must have OpenCV and Libtorch in your path or specify them in Cmakelists file 

## Run inference

```bash
./ts_ssd -s=path-to-model/your-ts-model.pt -b=backbone-network(i.e. mb1 or vgg16) -l=path-to-labels/labels.txt -p=0.5 -v=path-to-video/video
```
