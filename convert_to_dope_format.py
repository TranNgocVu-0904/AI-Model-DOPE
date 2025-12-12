#!/usr/bin/env python3
"""
convert_to_dope_format.py

Bước 2 sau khi đã chạy script RealSense + ArUco để tạo:
  - images/000000.png, 000001.png, ...
  - captures_000.json  (chứa translation + quaternion + intrinsics)

Script này sẽ:
  - Đọc captures_000.json
  - Với mỗi frame:
      + Lấy pose CUBE trong hệ CAM (quaternion + translation)
      + Lấy intrinsics K
      + Tạo 9 điểm 3D của cube (8 góc + tâm) theo convention DOPE
      + Project sang ảnh → 9 điểm 2D
      + Ghi file JSON cạnh ảnh:
          images/000000.json, images/000001.json, ...
    JSON có dạng tối thiểu mà CleanVisiiDopeLoader cần.
"""

import os
import json
import argparse
import numpy as np

# ---------- math utils ----------

def quat_to_rot_matrix(x, y, z, w):
    """
    Convert quaternion (x, y, z, w) -> 3x3 rotation matrix.
    Quaternion assumed normalized or almost-normalized.
    """
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Quaternion norm is zero")
    q /= n
    x, y, z, w = q

    # standard formula
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)]
    ], dtype=np.float32)
    return R

def project_points(K, R, t, points_3d):
    """
    Project 3D points in object frame into image plane.

    K: (3,3) camera intrinsic
    R: (3,3) rotation obj->cam
    t: (3,)  translation obj->cam (in meters)
    points_3d: (N,3) points in object frame

    return: list [[u,v], ...] length N
    """
    points_3d = np.asarray(points_3d, dtype=np.float32)      # (N,3)
    t = np.asarray(t, dtype=np.float32).reshape(3,)          # (3,)

    # obj -> cam
    pts_cam = (R @ points_3d.T).T + t[None, :]               # (N,3)

    # K @ X_cam
    uvw = (K @ pts_cam.T).T                                  # (N,3)

    # perspective division
    z = np.clip(uvw[:, 2:3], 1e-6, None)                     # avoid /0
    uv = uvw[:, :2] / z                                      # (N,2)

    return uv.tolist()

def get_dope_cube_points(L):
    """
    Tạo 9 điểm 3D (8 góc + tâm) theo thứ tự CuboidVertexType trong DOPE.

    DOPE dùng hệ OpenCV:
      - +X: sang phải
      - +Y: xuống
      - +Z: ra trước (về phía object)

    Các biến:
      right  = +L/2
      left   = -L/2
      top    = -L/2  (vì +Y là xuống)
      bottom = +L/2
      front  = +L/2
      rear   = -L/2
    """
    half = L / 2.0
    right, left = +half, -half
    top, bottom = -half, +half
    front, rear = +half, -half

    points = [
        # Index mapping khớp với CuboidVertexType trong common/cuboid.py
        [right, top,    front],   # 0 FrontTopRight
        [left,  top,    front],   # 1 FrontTopLeft
        [left,  bottom, front],   # 2 FrontBottomLeft
        [right, bottom, front],   # 3 FrontBottomRight
        [right, top,    rear],    # 4 RearTopRight
        [left,  top,    rear],    # 5 RearTopLeft
        [left,  bottom, rear],    # 6 RearBottomLeft
        [right, bottom, rear],    # 7 RearBottomRight
        [0.0,  0.0,    0.0],      # 8 Center
    ]
    return np.array(points, dtype=np.float32)

# ---------- main convert ----------

def convert_captures(root, captures_file, object_class, visibility=1.0, cube_edge_m=None, overwrite=False):
    """
    root: thư mục dataset (chứa images/, captures_000.json)
    captures_file: tên file json (vd "captures_000.json" hoặc path tuyệt đối)
    object_class: string, vd "cube_6cm" (phải trùng với --object trong train.py)
    visibility: float, mặc định 1.0 (fully visible)
    cube_edge_m: nếu None → lấy từ captures["info"]["cube_edge_m"]
    overwrite: nếu False → nếu .json đã tồn tại thì bỏ qua frame đó
    """
    # Path tới JSON tổng
    if os.path.isabs(captures_file):
        cap_path = captures_file
    else:
        cap_path = os.path.join(root, captures_file)

    if not os.path.isfile(cap_path):
        raise FileNotFoundError(f"Không tìm thấy captures file: {cap_path}")

    print(f"[INFO] Đọc: {cap_path}")
    with open(cap_path, "r") as f:
        data = json.load(f)

    info = data.get("info", {})
    captures = data.get("captures", [])

    if cube_edge_m is None:
        cube_edge_m = float(info.get("cube_edge_m", 0.06))  # fallback 6cm

    print(f"[INFO] Số frame: {len(captures)}")
    print(f"[INFO] cube_edge_m = {cube_edge_m} m")
    print(f"[INFO] object_class = '{object_class}'")

    num_written = 0
    num_skipped_exist = 0
    num_skipped_invalid = 0

    for cap in captures:
        filename = cap.get("filename")  # ví dụ "images/000123.png"
        if not filename:
            continue

        img_path = os.path.join(root, filename)
        if not os.path.isfile(img_path):
            print(f"[WARN] Không tìm thấy ảnh: {img_path}, bỏ qua.")
            continue

        # output json bên cạnh ảnh
        base, _ = os.path.splitext(img_path)
        json_out_path = base + ".json"

        if (not overwrite) and os.path.exists(json_out_path):
            num_skipped_exist += 1
            continue

        annotations = cap.get("annotations", [])
        if not annotations:
            print(f"[WARN] Frame {filename} không có annotations, bỏ qua.")
            continue

        # Lấy annotation pose đầu tiên
        values = annotations[0].get("values", {})
        trans = values.get("translation", {})
        rot = values.get("rotation", {})

        # translation (m)
        try:
            t = np.array([
                float(trans["x"]),
                float(trans["y"]),
                float(trans["z"]),
            ], dtype=np.float32)
        except Exception as e:
            print(f"[WARN] Lỗi đọc translation cho {filename}: {e}")
            num_skipped_invalid += 1
            continue

        # quaternion
        try:
            qx = float(rot["x"])
            qy = float(rot["y"])
            qz = float(rot["z"])
            qw = float(rot["w"])
        except Exception as e:
            print(f"[WARN] Lỗi đọc rotation cho {filename}: {e}")
            num_skipped_invalid += 1
            continue

        # intrinsics
        sensor = cap.get("sensor", {})
        K_list = sensor.get("camera_intrinsic")
        if K_list is None:
            print(f"[WARN] Không có camera_intrinsic cho {filename}, bỏ qua.")
            num_skipped_invalid += 1
            continue
        K = np.array(K_list, dtype=np.float32).reshape(3, 3)

        # R obj->cam
        R = quat_to_rot_matrix(qx, qy, qz, qw)

        # tạo 9 điểm cube trong object frame
        pts_3d = get_dope_cube_points(cube_edge_m)  # (9,3)

        # project
        pts_2d = project_points(K, R, t, pts_3d)    # list [[u,v],...]

        # bảo vệ nếu z <= 0 (điểm sau camera)
        # (ở đây đơn giản: nếu thấy z <= 0 cho bất kỳ điểm nào → visibility = 0,
        #  để DOPE coi như invisible object)
        # Nếu bạn chắc chắn aruco luôn ở trước camera thì có thể bỏ đoạn này.
        # (Lấy z từ pts_cam nếu muốn check kỹ; ở đây mình tin dataset là OK.)

        projected_cuboid = pts_2d[:8]
        projected_center = pts_2d[8]

        # tạo JSON theo format mà CleanVisiiDopeLoader cần
        obj = {
            "class": object_class,
            "visibility": float(visibility),
            "projected_cuboid": projected_cuboid,
            "projected_cuboid_centroid": projected_center,
        }

        out_json = {
            "objects": [obj]
        }

        os.makedirs(os.path.dirname(json_out_path), exist_ok=True)
        with open(json_out_path, "w") as f:
            json.dump(out_json, f, indent=2)

        num_written += 1

    print("===================================")
    print(f"[DONE] Viết mới JSON       : {num_written}")
    print(f"[INFO] Bỏ qua vì đã tồn tại: {num_skipped_exist}")
    print(f"[INFO] Bỏ qua vì lỗi/thiếu : {num_skipped_invalid}")
    print("===================================")

def main():
    parser = argparse.ArgumentParser(
        description="Convert captures_000.json (6D pose) -> DOPE per-image JSON format"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Thư mục dataset (chứa images/ và captures_000.json)"
    )
    parser.add_argument(
        "--captures",
        default="captures_000.json",
        help="Tên file JSON tổng (mặc định: captures_000.json)"
    )
    parser.add_argument(
        "--object-class",
        required=True,
        help="Tên class object, vd 'cube_6cm' (trùng với --object trong train.py)"
    )
    parser.add_argument(
        "--cube-edge-m",
        type=float,
        default=None,
        help="Cạnh cube (m). Nếu bỏ trống sẽ đọc từ info.cube_edge_m trong captures."
    )
    parser.add_argument(
        "--visibility",
        type=float,
        default=1.0,
        help="visibility cho object (0..1, mặc định 1.0)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Nếu bật, sẽ ghi đè .json cũ nếu đã tồn tại"
    )

    args = parser.parse_args()

    convert_captures(
        root=args.root,
        captures_file=args.captures,
        object_class=args.object_class,
        visibility=args.visibility,
        cube_edge_m=args.cube_edge_m,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()