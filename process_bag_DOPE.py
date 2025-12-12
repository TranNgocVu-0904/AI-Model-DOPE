#!/usr/bin/env python3
# Requires:
#   pip install opencv-contrib-python pyrealsense2 numpy

import os, json, argparse, math
import numpy as np
import cv2
import pyrealsense2 as rs

# ---------------- math utils ----------------

def frame_to_bgr(cf):
    """Trả về ảnh BGR đúng chuẩn để dùng với OpenCV (vẽ & imwrite)."""
    arr = np.asanyarray(cf.get_data())
    fmt = cf.get_profile().format()
    if fmt == rs.format.rgb8:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if fmt == rs.format.bgr8:
        return arr
    if fmt in (rs.format.yuyv, rs.format.yuy2):
        return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
    if fmt == rs.format.uyvy:
        return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_UYVY)
    # fallback: coi như RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], np.float32)

def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], np.float32)

def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],
                     [s, c,0],
                     [0, 0,1]], np.float32)

def rpy_to_R(rx, ry, rz):
    # roll(x), pitch(y), yaw(z)
    return rotz(rz) @ roty(ry) @ rotx(rx)

def mat_to_quat(R):
    # 3x3 rotation -> (x,y,z,w) normalized
    m = R.astype(np.float64)
    t = np.trace(m)
    if t > 0.0:
        r = math.sqrt(1.0 + t); w = 0.5 * r; r = 0.5 / r
        x = (m[2,1] - m[1,2]) * r
        y = (m[0,2] - m[2,0]) * r
        z = (m[1,0] - m[0,1]) * r
    else:
        i = int(np.argmax([m[0,0], m[1,1], m[2,2]]))
        if i == 0:
            r = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]); x = 0.5 * r; r = 0.5 / r
            y = (m[0,1] + m[1,0]) * r; z = (m[0,2] + m[2,0]) * r; w = (m[2,1] - m[1,2]) * r
        elif i == 1:
            r = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]); y = 0.5 * r; r = 0.5 / r
            x = (m[0,1] + m[1,0]) * r; z = (m[1,2] + m[2,1]) * r; w = (m[0,2] - m[2,0]) * r
        else:
            r = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]); z = 0.5 * r; r = 0.5 / r
            x = (m[0,2] + m[2,0]) * r; y = (m[1,2] + m[2,1]) * r; w = (m[1,0] - m[0,1]) * r
    q = np.array([x, y, z, w], np.float32)
    return q / np.linalg.norm(q)

# --------- drawing & geometry helpers ---------

def _cube_points_8(L):
    """8 góc cube, tâm tại (0,0,0)."""
    base = np.array([
        [-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5],
        [-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5],
    ], np.float32)
    return base * L  # (8,3)

def _cube_points_9(L):
    """8 góc + tâm cube."""
    corners = _cube_points_8(L)
    center = np.zeros((1,3), np.float32)
    return np.vstack([corners, center])  # (9,3)

def _project(K, R, t, X):
    # K:(3,3); R:(3,3); t:(3,); X:(N,3) -> (N,2)
    Xc = (R @ X.T).T + t[None, :]
    uvw = (K @ Xc.T).T
    return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)

def corner_area(corner):
    pts = np.array(corner).reshape(-1, 2)
    x = pts[:, 0]; y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

# ---------------- ArUco utils ----------------

def parse_dict(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco không có. Cần cài opencv-contrib-python.")
    name = name.strip().upper()
    alias = {
        "ORIGINAL": "DICT_ARUCO_ORIGINAL",
        "ARUCO_ORIGINAL": "DICT_ARUCO_ORIGINAL",
        "MIP_36H12": "DICT_ARUCO_MIP_36h12",
        "APRILTAG_16H5": "DICT_APRILTAG_16h5",
        "APRILTAG_25H9": "DICT_APRILTAG_25h9",
        "APRILTAG_36H10": "DICT_APRILTAG_36h10",
        "APRILTAG_36H11": "DICT_APRILTAG_36h11",
    }
    if name in alias:
        const_name = alias[name]
    else:
        const_name = name if name.startswith("DICT_") else "DICT_" + name
    if not hasattr(cv2.aruco, const_name):
        available = [k for k in dir(cv2.aruco) if k.startswith("DICT_")]
        raise ValueError(f"Unknown dict '{name}'. Available:\n  " + "\n  ".join(sorted(available)))
    return getattr(cv2.aruco, const_name)

def create_detector(dict_name: str):
    dict_id = parse_dict(dict_name)
    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV mới
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        def detect(img_gray):
            return detector.detectMarkers(img_gray)  # corners(list), ids(np Nx1), rejected
    else:  # OpenCV cũ
        aruco_dict = cv2.aruco.Dictionary_get(dict_id)
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        def detect(img_gray):
            corners, ids, rejected = cv2.aruco.detectMarkers(img_gray, aruco_dict, parameters=params)
            return corners, ids, rejected
    return detect

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Extract cube GT (single ArUco on TOP face).")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dict", required=True, help="e.g. 4X4_250, 4X4_1000, ARUCO_ORIGINAL, MIP_36H12")
    ap.add_argument("--marker-id", type=int, required=True, help="ID tag dán trên cube (duy nhất).")
    ap.add_argument("--marker-len-m", type=float, required=True, help="Cạnh đen của tag (m).")
    ap.add_argument("--cube-edge-m", type=float, required=True, help="Cạnh cube (m).")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = lấy hết.")
    ap.add_argument("--tag-rpy-deg", default="0,0,0", help="Tag->Cube (roll,pitch,yaw deg).")
    args = ap.parse_args()

    if args.marker_len_m <= 0 or args.cube_edge_m <= 0:
        raise ValueError("marker-len-m và cube-edge-m phải > 0")

    os.makedirs(args.out, exist_ok=True)
    img_raw_dir = os.path.join(args.out, "images")
    os.makedirs(img_raw_dir, exist_ok=True)

    # RealSense playback
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(args.bag, repeat_playback=False)
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)

    # Color intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    W, H = intr.width, intr.height
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    K0 = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0,  0,  1]], np.float32)
    dist = np.array(intr.coeffs, np.float32).reshape(-1)
    dist_use = None if dist.size == 0 or np.allclose(dist, 0) else dist

    detect = create_detector(args.dict)

    # Tag->Cube: tag là mặt TRÊN, Z_tag // +Z_cube
    L = float(args.cube_edge_m)
    rx_deg, ry_deg, rz_deg = [float(x) for x in args.tag_rpy_deg.split(",")]
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    R_tag_obj = rpy_to_R(rx, ry, rz).astype(np.float32)
    # tâm CUBE nằm dọc −Z_tag
    t_tag_obj = np.array([0.0, 0.0, -L/2.0], np.float32)

    records = []
    idx = 0
    kept = 0
    used_frames = 0

    print("[INFO] --- CONFIG ---")
    print(f"bag: {args.bag}")
    print(f"dict: {args.dict} | marker_id: {args.marker_id}")
    print(f"marker_len_m: {args.marker_len_m} | cube_edge_m: {L}")
    print(f"tag_rpy_deg: ({rx_deg}, {ry_deg}, {rz_deg})")
    print("---------------------")

    try:
        while True:
            try:
                frames = pipe.wait_for_frames()
            except RuntimeError:
                print("[INFO] End of bag")
                break

            cf = frames.get_color_frame()
            if not cf:
                idx += 1
                continue

            img = frame_to_bgr(cf)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detect(gray)
            if ids is None or len(ids) == 0:
                idx += 1
                continue

            # lọc đúng ID & chọn cái to nhất
            candidate = []
            for c, mid in zip(corners, ids.flatten()):
                if int(mid) == args.marker_id:
                    candidate.append((corner_area(c), c))
            if not candidate:
                idx += 1
                continue
            candidate.sort(key=lambda x: -x[0])
            _, c_best = candidate[0]

            # pose tag (giữ dạng list như detect trả ra để API ổn định)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [c_best], args.marker_len_m, K0, dist_use
            )
            rvec = np.array(rvecs).reshape(-1,3)[0]
            tvec = np.array(tvecs).reshape(-1,3)[0].astype(np.float32)
            R_cam_tag, _ = cv2.Rodrigues(rvec)

            # ÉP: Z_tag phải hướng VỀ camera (chuẩn quy ước)
            z_cam = R_cam_tag @ np.array([0,0,1], np.float32)  # Z_tag trong hệ CAM
            # vector CAM->TAG là tvec; nếu cùng hướng => Z_tag đang hướng ra xa camera -> lật 180° quanh X_tag
            if float(np.dot(z_cam, tvec)) > 0:
                R_flip = np.array([[1,0,0],[0,-1,0],[0,0,-1]], np.float32)  # Rx(pi)
                R_cam_tag = (R_cam_tag @ R_flip).astype(np.float32)

            # cam <- obj
            R_cam_obj = (R_cam_tag @ R_tag_obj).astype(np.float32)
            t_cam_obj = (R_cam_tag @ t_tag_obj + tvec).astype(np.float32)

            # đặt tên file cho frame
            base = f"{idx:06d}.png"

            # 1) chỉ lưu ảnh gốc
            cv2.imwrite(os.path.join(img_raw_dir, base), img)

            # 2) Keypoints cho DOPE-style: 8 góc + tâm
            keypoints_3d = _cube_points_9(L)                       # (9,3)
            keypoints_2d = _project(K0, R_cam_obj, t_cam_obj,
                                    keypoints_3d)                  # (9,2)

            # JSON annotation (pose CUBE trong hệ CAM)
            q = mat_to_quat(R_cam_obj)
            records.append({
                "filename": f"images/{base}",
                "annotations": [{
                    "task": "pose",
                    "values": {
                        "object_class": "cube_%.0fmm" % (L*1000.0),
                        "tag_id": int(args.marker_id),
                        "translation": {
                            "x": float(t_cam_obj[0]),
                            "y": float(t_cam_obj[1]),
                            "z": float(t_cam_obj[2])
                        },
                        "rotation": {
                            "x": float(q[0]),
                            "y": float(q[1]),
                            "z": float(q[2]),
                            "w": float(q[3])
                        },
                        "size": {
                            "x": L, "y": L, "z": L
                        },
                        # 9 điểm: 8 góc + tâm (theo cùng thứ tự luôn, frame tâm cube)
                        "keypoints_3d": keypoints_3d.tolist(),   # 9 x [X,Y,Z]
                        "keypoints_2d": keypoints_2d.tolist(),   # 9 x [u,v]
                    }
                }],
                "sensor": {
                    "camera_intrinsic": K0.tolist()
                }
            })

            kept += 1
            idx  += 1
            used_frames += 1
            if args.max_frames > 0 and kept >= args.max_frames:
                print(f"[INFO] Reached max-frames = {args.max_frames}")
                break

    finally:
        pipe.stop()

    out_json = {
        "info": {
            "note": "Cube GT from single ArUco on TOP face; Z_tag -> camera; cube center along -Z_tag.",
            "image_width": int(W),
            "image_height": int(H),
            "aruco_dict": args.dict,
            "marker_id": int(args.marker_id),
            "marker_len_m": float(args.marker_len_m),
            "cube_edge_m": float(L),
            "tag_rpy_deg": [rx_deg, ry_deg, rz_deg],
            "frames_used": int(used_frames),
            "extra_folders": {
                "images": "images/"
            }
        },
        "captures": records
    }

    json_path = os.path.join(args.out, "captures_000.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"[DONE] raw images : {img_raw_dir}")
    print(f"[DONE] JSON       : {json_path}")

if __name__ == "__main__":
    main()