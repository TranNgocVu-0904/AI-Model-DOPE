 # AI-Model-Using-DOPE


```bash
git clone https://github.com/TranNgocVu-0904/AI-Model-Using-DOPE.git
```


## Realtime với Intel RealSense

### Cài requirements (bỏ mấy cái không cần nếu lỗi)


```bash
cd ~/AI-Model-Using-DOPE/Deep_Object_Pose
pip install -r requirements.txt
```
Hạ NumPy xuống < 2

```bash
pip uninstall -y numpy

pip install "numpy<2,>=1.26"   # sẽ cài 1.26.4 cho Python 3.12
```
Hạ  Open-CV xuống: 

```bash
pip install --force-reinstall "opencv-python==4.6.0.66"
```

### Cài thư viện cần thiết:

```bash
pip install opencv-python
pip install pyrealsense2           # cần cho camera RealSense
```
</pre>





### Chạy realtime

Nếu muốn chỉnh sửa file realtime_dope_realsense.py (lấy tọa độ realtime) để thêm socket và địa chỉ IP:

```bash
cd ~/AI-Model-Using-DOPE/Deep_Object_Pose/realtime
open realtime_dope_realsense.py
```
rồi chỉnh sửa

Từ root repository:


```bash
cd ~/AI-Model-Using-DOPE/Deep_Object_Pose
```

Cần có file final_net_cube_6cm_0030.pth để chạy được DOPE model:

```bash
[File Model DOPE AI](https://drive.google.com/file/d/1fKnnWy36cJvyrl-w-5_T44c_w5oWR4Fu/view usp=sharing)
```
Tải về rồi cho vào thư mục:

```bash
~/AI-Model-Using-DOPE/output_cube_6cm/
```

Nếu weight được train bằng DDP (state_dict có prefix "module.") thì thêm --parallel:


```bash

python realtime/realtime_dope_realsense.py --weights /AI-Model-Using-DOPE/output_cube_6cm/final_net_cube_6cm_0030.pth --config config/config_pose_cube_6cm.yaml --object cube_6cm --parallel

```
