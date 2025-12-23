# AI-Model-DOPE

```bash
git clone https://github.com/TranNgocVu-0904/AI-Model-DOPE.git
```

## Note about GPU (CUDA) vs CPU mode

**GPU (CUDA) support**: If you have an **NVIDIA GPU**, you can run **DOPE** with **CUDA** for significantly faster inference.

**CPU support (modified)**: In this repository, we modified the original upstream source code so that the realtime pipeline can also run on **CPU-only machines** (without **CUDA**). This is useful for users who do not have an **NVIDIA GPU**. Obviously, performance will not be as good as when running on NVIDIA GPU.

**MacOS users**: Since the RealSense + ROS/DOPE pipeline is typically developed and tested on **Linux**, MacOS users are recommended to run the project inside an **Ubuntu Virtual Machine** (e.g., UTM/VMware/VirtualBox). With an **Ubuntu VM**, you can run the CPU version and reproduce the same environment as on Linux.
## Realtime detection with Intel RealSense

### Install requirements (remove unnecessary ones if errors occur)

```bash
cd AI-Model-DOPE/Deep_Object_Pose
pip install -r requirements.txt
```
Reduce NumPy to < 2

```bash
pip uninstall -y numpy

pip install "numpy<2,>=1.26" # will install 1.26.4 for Python 3.12

```
Reduce Open-CV to:

```bash
pip install --force-reinstall "opencv-python==4.6.0.66"

```
### Install necessary libraries Instructions:

```bash
pip install opencv-python
pip install pyrealsense2 # needed for RealSense camera
```
</pre>

From root repository (AI-Model-DOPE):

```bash
cd Deep-Object-Pose
```

The final_net_cube_6cm_0030.pth file is needed to run the DOPE model: [Download DOPE AI Model File](https://drive.google.com/drive/folders/1wQ5EPUMy37dbofn4g7YrulYI6OAYfwDT?usp=sharing)

Download and place it in the folder:

```bash
weights
```

If weights are trained using DDP (state_dict has the prefix "module."), add --parallel:

```bash

python realtime/realtime_dope_realsense.py --weights weights/final_net_cube_6cm_0030.pth --config config/config_pose_cube_6cm.yaml --object cube_6cm --parallel

```
## About the dataset

We provide a small sample of the dataset we used for training, including **100 RGB images** and **100 corresponding JSON annotation files** (one JSON per image) following the DOPE cuboid annotation format.

Please note that this sample is only meant for **reference and demonstration**. In practice, training a DOPE model typically requires **at least ~10,000 images** and **~10,000 matching JSON annotations** to achieve stable and accurate performance.