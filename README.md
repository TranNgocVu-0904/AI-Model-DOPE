# AI-Model-DOPE

```bash
git clone https://github.com/TranNgocVu-0904/AI-Model-DOPE.git
```

## Realtime with Intel RealSense

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

The final_net_cube_6cm_0030.pth file is needed to run the DOPE model: [Download DOPE AI Model File](https://drive.google.com/file/d/1fKnnWy36cJvyrl-w-5_T44c_w5oWR4Fu/view?usp=sharing)

Download and place it in the folder:

```bash
weights
```

If weights are trained using DDP (state_dict has the prefix "module."), add --parallel:

```bash

python realtime/realtime_dope_realsense.py --weights /weights/final_net_cube_6cm_0030.pth --config config/config_pose_cube_6cm.yaml --object cube_6cm --parallel

```