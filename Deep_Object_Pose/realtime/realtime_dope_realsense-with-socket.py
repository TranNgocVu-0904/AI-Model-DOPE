#!/usr/bin/env python3
# Realtime 6D pose with DOPE + Intel RealSense + Socket sender
# deps: pip install opencv-contrib-python pyrealsense2 pillow numpy pyyaml simplejson

import argparse
import time
import math
import sys
import os
import socket   # <-- thêm cho socket

import numpy as np
import cv2
from PIL import Image
import pyrealsense2 as rs
import yaml

# ==== SOCKET CONFIG (Device 2 / ROS bridge) ====
ROS_IP = '172.16.130.140'  # IP máy nhận (chạy dope_bridge.py hoặc ROS-node)
ROS_PORT = 5005            # Port phải trùng với bridge/server
SEND_DELAY = 0.5           # Gửi tối đa 2 lần/giây để đỡ lag
last_sent_time = 0.0
# ==============================================

# ==== import từ DOPE common ====
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from utils import Draw


# ===== Helper: gửi pose qua socket =====
def send_to_ros(location, quaternion):
    """
    Gửi pose dạng: x,y,z,qx,qy,qz,qw qua TCP socket tới ROS bridge.
    location: np.array hoặc list [x, y, z]
    quaternion: np.array hoặc list [qx, qy, qz, qw] (hoặc [w,x,y,z] tuỳ DOPE, bạn tự thống nhất bên receiver)
    """
    global last_sent_time

    now = time.time()
    if now - last_sent_time < SEND_DELAY:
        return  # chưa đủ thời gian, skip frame này

    try:
        msg = (
            f"{location[0]:.4f},{location[1]:.4f},{location[2]:.4f},"
            f"{quaternion[0]:.4f},{quaternion[1]:.4f},"
            f"{quaternion[2]:.4f},{quaternion[3]:.4f}"
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)  # timeout nhanh để không treo frame
            s.connect((ROS_IP, ROS_PORT))
            s.sendall(msg.encode("utf-8"))

        print(f"[NET] Sent to ROS: {msg}")
        last_sent_time = now

    except ConnectionRefusedError:
        print(f"[NET] Connection refused: server {ROS_IP}:{ROS_PORT} chưa chạy?")
    except socket.timeout:
        print(f"[NET] Error: timed out khi gửi tới {ROS_IP}:{ROS_PORT}")
    except Exception as e:
        print(f"[NET] Error: {e}")


# ========================================


class DopeRealtime:
    def __init__(self, config_path, weight_path, class_name, parallel=False):
        # ----- load config -----
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.input_is_rectified = True  # RealSense cho mình K đã rectified
        self.downscale_height = config["downscale_height"]

        # config_detect giống inference.py
        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config["thresh_angle"]
        self.config_detect.thresh_map = config["thresh_map"]
        self.config_detect.sigma = config["sigma"]
        self.config_detect.thresh_points = config["thresh_points"]

        # ----- load model DOPE -----
        self.model = ModelData(
            name=class_name,
            net_path=weight_path,
            parallel=parallel,
        )
        t0 = time.time()
        self.model.load_net_model()
        print(f"DOPE model loaded in {time.time() - t0:.2f} s")

        self.class_name = class_name

        # màu vẽ cube (nếu không có trong config thì mặc định xanh lá)
        try:
            self.draw_color = tuple(config["draw_colors"][class_name])
        except Exception:
            self.draw_color = (0, 255, 0)

        self.dimension = tuple(config["dimensions"][class_name])
        self.class_id = config["class_ids"][class_name]

        self.pnp_solver = CuboidPNPSolver(
            class_name,
            cuboid3d=Cuboid3d(config["dimensions"][class_name]),
        )

        print("Ctrl-C / ESC / q để thoát.")

    def infer(self, frame_bgr, K_3x3):
        """
        frame_bgr: (H,W,3) BGR từ RealSense
        K_3x3: ma trận nội suy 3x3 (fx,fy,cx,cy) của camera
        return:
          - results: list dict DOPE (location, quaternion, projected_points, ...)
          - vis_bgr: ảnh có vẽ cube (BGR) để hiển thị
        """
        # BGR -> RGB (DOPE dùng RGB)
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        K = np.array(K_3x3, dtype="float64").copy()

        # downscale giống inference.py
        scaling_factor = float(self.downscale_height) / float(h)
        if scaling_factor < 1.0:
            K[0, :] *= scaling_factor
            K[1, :] *= scaling_factor
            img_rgb = cv2.resize(
                img_rgb,
                (int(w * scaling_factor), int(h * scaling_factor)),
                interpolation=cv2.INTER_LINEAR,
            )

        self.pnp_solver.set_camera_intrinsic_matrix(K)
        # RealSense mình bỏ qua distortion (hoặc dùng intr.coeffs nếu muốn chính xác)
        self.pnp_solver.set_dist_coeffs(np.zeros((4, 1), dtype="float64"))

        # copy để vẽ
        img_copy = img_rgb.copy()
        pil_im = Image.fromarray(img_copy)
        draw = Draw(pil_im)

        results, belief_imgs = ObjectDetector.detect_object_in_image(
            self.model.net,
            self.pnp_solver,
            img_rgb,
            self.config_detect,
            grid_belief_debug=False,
        )

        good_results = []

        for r in results:
            if r["location"] is None:
                continue
            proj = r["projected_points"]
            if proj is None or any(p is None for p in proj):
                # incomplete cuboid
                continue

            # vẽ cuboid 2D
            pts2d = [tuple(p) for p in proj]
            draw.draw_cube(pts2d, self.draw_color)
            good_results.append(r)

        vis_rgb = np.array(pil_im)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        return good_results, vis_bgr


def start_realsense(width=640, height=480, fps=30):
    """Khởi động RealSense, trả về (pipeline, intrinsics K)."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype="float64",
    )

    print("RealSense intrinsics:")
    print(K)
    return pipeline, K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        "-w",
        required=True,
        help="Path tới file DOPE weight (.pth), ví dụ ../output_cube_6cm/final_net_cube_6cm_0030.pth",
    )
    parser.add_argument(
        "--config",
        default="../config/config_pose.yaml",
        help="Path tới config_pose.yaml (mặc định ../config/config_pose.yaml)",
    )
    parser.add_argument(
        "--object",
        required=True,
        help="Tên class trong config, ví dụ cube_6cm",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Đặt nếu weight được train bằng DDP (có prefix 'module.').",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Chiều rộng stream màu RealSense (default 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Chiều cao stream màu RealSense (default 480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS của camera (default 30)",
    )

    args = parser.parse_args()

    # --- khởi động RealSense ---
    pipeline, K = start_realsense(args.width, args.height, args.fps)

    # --- load DOPE ---
    dope = DopeRealtime(
        config_path=args.config,
        weight_path=args.weights,
        class_name=args.object,
        parallel=args.parallel,
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())

            results, vis_bgr = dope.infer(color_bgr, K)

            # Lấy detection tốt nhất (nếu có)
            if results:
                r0 = results[0]
                loc = r0["location"]          # [x, y, z] (m)
                quat = r0["quaternion"]       # [x, y, z, w] hoặc tương tự (xyzw)

                # 1) In ra tọa độ như file đầu
                print(
                    "Pose:",
                    "x={:.3f} m".format(loc[0]),
                    "y={:.3f} m".format(loc[1]),
                    "z={:.3f} m".format(loc[2]),
                    "| quat =", ["{:.3f}".format(q) for q in quat],
                )

                # 2) Gửi qua socket như file thứ hai
                send_to_ros(loc, quat)

                # Vẽ text lên frame
                cv2.putText(
                    vis_bgr,
                    f"x={loc[0]:.2f} y={loc[1]:.2f} z={loc[2]:.2f} (m)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("DOPE Realsense", vis_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC hoặc q
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()