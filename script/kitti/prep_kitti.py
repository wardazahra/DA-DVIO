#!/usr/bin/env python3
import os, sys, glob, argparse, shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R

R_EARTH = 6378137.0

# ---------------------------- utils ----------------------------

def read_lines(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines()]

def parse_kitti_timestamp_line(s):
    # Example: 2011-09-30 18:17:33.028161000
    if "." in s:
        date_part, frac = s.split(".")
        frac = (frac + "000000000")[:9]  # nanoseconds to 9 digits
        dt = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp() + int(frac) * 1e-9
    else:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()

def load_timestamps_file(path_txt):
    lines = read_lines(path_txt)
    return np.array([parse_kitti_timestamp_line(s) for s in lines], dtype=float)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _read_kv_file(path):
    kv = {}
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            if ":" in ln:
                k, v = ln.split(":", 1)
                kv[k.strip()] = v.strip()
    return kv

def _parse_vec(s):
    return np.array(list(map(float, s.split())))

def to44(R3, t3):
    T = np.eye(4)
    T[:3,:3] = R3
    T[:3, 3] = t3.reshape(3)
    return T

# ---------------------- calibration loaders --------------------

def load_T_imu_to_velo(calib_root):
    p = Path(calib_root) / "calib_imu_to_velo.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    kv = _read_kv_file(p)
    R_ = _parse_vec(kv["R"]).reshape(3,3)
    T_ = _parse_vec(kv["T"]).reshape(3)
    return to44(R_, T_)           # T_i^velo

def load_T_velo_to_cam0(calib_root):
    p = Path(calib_root) / "calib_velo_to_cam.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    kv = _read_kv_file(p)
    R_ = _parse_vec(kv["R"]).reshape(3,3)
    T_ = _parse_vec(kv["T"]).reshape(3)
    return to44(R_, T_)           # T_velo^cam0 (raw)

def load_cam_rect_and_proj(calib_root):
    # calib_cam_to_cam.txt may be in drive_root or its parent date folder
    p = Path(calib_root) / "calib_cam_to_cam.txt"
    if not p.exists():
        p2 = Path(calib_root).parent / "calib_cam_to_cam.txt"
        if not p2.exists():
            raise FileNotFoundError(f"Missing {p} (and {p2})")
        p = p2
    kv = _read_kv_file(p)

    def mat_from_key(k, shape):
        if k not in kv:
            raise KeyError(f"{k} not found in {p}")
        return _parse_vec(kv[k]).reshape(shape)

    R_rect_00 = mat_from_key("R_rect_00", (3,3))
    P_rect_00 = mat_from_key("P_rect_00", (3,4))
    P_rect_02 = mat_from_key("P_rect_02", (3,4))
    return R_rect_00, P_rect_00, P_rect_02

def compute_T_cam0rect_to_cam2rect(P_rect_00, P_rect_02):
    # In rectified space, relative rotation cam0->cam2 is identity; only baseline along +x
    fx = P_rect_00[0,0]
    b  = -(P_rect_02[0,3] - P_rect_00[0,3]) / fx
    T  = np.eye(4)
    T[0,3] = b
    return T  # T_cam0rect^cam2rect

def compute_T_i_to_c2(calib_root):
    T_i_velo     = load_T_imu_to_velo(calib_root)     # IMU -> Velodyne
    T_velo_cam0  = load_T_velo_to_cam0(calib_root)    # Velodyne -> Cam0 (raw)
    R_rect_00, P_rect_00, P_rect_02 = load_cam_rect_and_proj(calib_root)

    T_rect00 = np.eye(4); T_rect00[:3,:3] = R_rect_00
    T_cam0rect_cam2rect = compute_T_cam0rect_to_cam2rect(P_rect_00, P_rect_02)

    # Velodyne -> Cam2 (rectified)
    T_velo_cam0rect = T_rect00 @ T_velo_cam0
    T_velo_cam2rect = T_cam0rect_cam2rect @ T_velo_cam0rect

    # IMU -> Cam2 (rectified)
    T_i_cam2 = T_velo_cam2rect @ T_i_velo
    return T_i_cam2, P_rect_02

# --------------------- OXTS / trajectory ----------------------

def latlon_to_mercator(lat_deg, lon_deg, scale_lat0_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    scale = np.cos(np.deg2rad(scale_lat0_deg))
    mx = scale * R_EARTH * lon
    my = scale * R_EARTH * np.log(np.tan(np.pi/4.0 + lat/2.0))
    return mx, my

def load_oxts_packets(oxts_dir):
    files = sorted(glob.glob(str(Path(oxts_dir) / "data" / "*.txt")))
    if not files:
        raise FileNotFoundError(f"No OXTS data files in {oxts_dir}/data")
    packs = []
    for fp in files:
        vals = list(map(float, read_lines(fp)[0].split()))
        packs.append(vals)
    return np.array(packs, dtype=float)

def build_T_w_imu_seq(oxts_packets):
    # Using KITTI convention: columns include lat lon alt roll pitch yaw (rad)
    lat0, lon0, alt0 = oxts_packets[0][0], oxts_packets[0][1], oxts_packets[0][2]
    mx0, my0 = latlon_to_mercator(lat0, lon0, lat0)

    T_w_i_list = []
    for p in oxts_packets:
        lat, lon, alt = p[0], p[1], p[2]
        roll, pitch, yaw = p[3], p[4], p[5]  # radians
        mx, my = latlon_to_mercator(lat, lon, lat0)
        t = np.array([mx - mx0, my - my0, alt - alt0])
        Rwi = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        T = np.eye(4); T[:3,:3] = Rwi; T[:3,3] = t
        T_w_i_list.append(T)
    return T_w_i_list

# ------------------------ writers ------------------------------

def write_timestamps_sec(out_path, ts_rel):
    with open(out_path, "w") as f:
        for t in ts_rel:
            f.write(f"{t:.9f}\n")

def write_calib_fx_fy_cx_cy(out_path, P_rect_02):
    fx, fy, cx, cy = P_rect_02[0,0], P_rect_02[1,1], P_rect_02[0,2], P_rect_02[1,2]
    np.savetxt(out_path, np.array([fx, fy, cx, cy]), fmt="%.9f")

def write_imu_csv(out_csv, imu_ts_rel, oxts_packets):
    # Accel m/s^2: cols 11,12,13 ; Gyro rad/s: cols 20,21,22 (KITTI OXTS spec)
    ax, ay, az = oxts_packets[:, 11], oxts_packets[:, 12], oxts_packets[:, 13]
    wx, wy, wz = oxts_packets[:, 20], oxts_packets[:, 21], oxts_packets[:, 22]
    with open(out_csv, "w") as f:
        f.write("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        for t, gx, gy, gz, ax_i, ay_i, az_i in zip(imu_ts_rel, wx, wy, wz, ax, ay, az):
            t_ns = int(round(t * 1e9))  # seconds -> ns
            f.write(f"{t_ns},{gx:.10f},{gy:.10f},{gz:.10f},{ax_i:.10f},{ay_i:.10f},{az_i:.10f}\n")

def write_gt_stamped_cam(out_path, img_ts_rel, T_w_c_list):
    with open(out_path, "w") as f:
        for t, T in zip(img_ts_rel, T_w_c_list):
            q = R.from_matrix(T[:3,:3]).as_quat()  # x y z w
            x, y, z = T[:3,3]
            f.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")

def write_yaml_hint(out_path, Ti1c):
    flat = ", ".join(f"{x:.9f}" for x in Ti1c.reshape(-1))
    with open(out_path, "w") as f:
        f.write("# Suggested runtime config bits matching the preprocessed outputs\n")
        f.write("IMU_TIME_UNIT: 'ns'\n")
        f.write("GYRO_UNIT: 'rad_s'\n")
        f.write("Ti1c: [" + flat + "]\n")

# ------------------------ main prep ----------------------------

def main():
    ap = argparse.ArgumentParser(description="KITTI Raw (synced+rectified) → VIO preprocessing")
    ap.add_argument("--drive_root", required=True,
                    help="e.g. /media/.../2011_09_30/2011_09_30_drive_0018_sync")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--copy_images", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()

    drive = Path(args.drive_root)
    img_dir  = drive / "image_02"
    oxts_dir = drive / "oxts"

    # Validate structure
    if not (img_dir / "data").exists(): raise FileNotFoundError(f"Missing {img_dir}/data")
    if not (img_dir / "timestamps.txt").exists(): raise FileNotFoundError(f"Missing {img_dir}/timestamps.txt")
    if not (oxts_dir / "data").exists(): raise FileNotFoundError(f"Missing {oxts_dir}/data")
    if not (oxts_dir / "timestamps.txt").exists(): raise FileNotFoundError(f"Missing {oxts_dir}/timestamps.txt")

    # Compute Ti1c (IMU→Cam2 rectified) and grab P_rect_02
    Ti1c, P_rect_02 = compute_T_i_to_c2(drive)
    print("\n[Ti1c] IMU→Cam2 (rectified):\n", np.array2string(Ti1c, precision=6, suppress_small=True))

    # Load timestamps
    img_ts_abs = load_timestamps_file(img_dir / "timestamps.txt")
    imu_ts_abs = load_timestamps_file(oxts_dir / "timestamps.txt")
    print(f"[i] #images={len(img_ts_abs)}  #imu={len(imu_ts_abs)}")

    # Load OXTS packets (one per imu timestamp)
    oxts_packets = load_oxts_packets(oxts_dir)
    if len(oxts_packets) != len(imu_ts_abs):
        n = min(len(oxts_packets), len(imu_ts_abs))
        print(f"[!] OXTS count ({len(oxts_packets)}) != imu ts ({len(imu_ts_abs)}); truncating to {n}")
        oxts_packets = oxts_packets[:n]
        imu_ts_abs   = imu_ts_abs[:n]

    # Overlap
    t0 = max(img_ts_abs[0], imu_ts_abs[0])
    t1 = min(img_ts_abs[-1], imu_ts_abs[-1])
    if t1 <= t0:
        raise RuntimeError("No temporal overlap between images and IMU.")
    img_mask = (img_ts_abs >= t0) & (img_ts_abs <= t1)
    imu_mask = (imu_ts_abs >= t0) & (imu_ts_abs <= t1)

    img_ts_abs = img_ts_abs[img_mask]
    imu_ts_abs = imu_ts_abs[imu_mask]
    oxts_packets = oxts_packets[imu_mask]

    # Relative times from overlap start
    img_ts_rel = img_ts_abs - t0
    imu_ts_rel = imu_ts_abs - t0

    print(f"[i] Overlap: {t0:.6f} .. {t1:.6f}  ({t1-t0:.2f} s)")
    print(f"[i] After sync: images={len(img_ts_rel)}  imu={len(imu_ts_rel)}")
    if len(img_ts_rel) == 0 or len(imu_ts_rel) == 0:
        raise RuntimeError("Empty overlap after sync.")

    # World→IMU poses from OXTS
    T_w_i_seq = build_T_w_imu_seq(oxts_packets)

    # Associate IMU-indexed poses to image timestamps by nearest IMU sample
    idx_for_img = np.searchsorted(imu_ts_rel, img_ts_rel, side='left')
    idx_for_img = np.clip(idx_for_img, 0, len(imu_ts_rel)-1)
    T_w_i_for_img = [T_w_i_seq[i] for i in idx_for_img]

    # Map to camera: T_w^c = T_w^i @ T_i^c
    T_w_c_for_img = [T_w_i @ Ti1c for T_w_i in T_w_i_for_img]

    # --------------- write outputs ----------------
    # --------------- write outputs ----------------
    out = Path(args.output_dir)
    ensure_dir(out)
    img_out_dir = out / "images_undistorted_left"
    ensure_dir(img_out_dir)

    # Save images: use CORRECT indices matching the overlap window
    src_imgs = sorted(glob.glob(str(img_dir / "data" / "*.png")))

    # Find which source images correspond to the overlap window
    original_indices = np.where(img_mask)[0]

    if len(original_indices) != len(img_ts_rel):
        print(f"[!] Index count mismatch: {len(original_indices)} vs {len(img_ts_rel)}")
        N = min(len(original_indices), len(img_ts_rel))
    else:
        N = len(original_indices)

    # CRITICAL: Trim IMU data to ensure it covers ALL images
    # Each image consumes ~2-3 IMU measurements in DBA
    imu_per_image = len(imu_ts_rel) / N
    min_imu_per_image = 3.0  # Safety margin

    if imu_per_image < min_imu_per_image:
        # Not enough IMU data - reduce image count
        max_safe_images = int(len(imu_ts_rel) / min_imu_per_image)
        print(f"[!] WARNING: Only {imu_per_image:.2f} IMU/image (need >{min_imu_per_image})")
        print(f"[!] Reducing from {N} to {max_safe_images} images to ensure IMU coverage")
        N = max_safe_images
        original_indices = original_indices[:N]
        img_ts_rel = img_ts_rel[:N]

    print(f"[i] Writing {N} images to {img_out_dir} …")
    for i in range(N):
        src = src_imgs[original_indices[i]]  # ← Use correct source index
        dst = img_out_dir / f"{i:012d}.png"
        if args.copy_images:
            shutil.copy2(src, dst)
        else:
            try:
                if dst.exists(): dst.unlink()
                os.symlink(Path(src), dst)
            except Exception:
                shutil.copy2(src, dst)

        # Re-associate poses for trimmed image set
        if N < len(T_w_c_for_img):
            T_w_c_for_img = T_w_c_for_img[:N]

        # timestamps (seconds, relative)
        write_timestamps_sec(out / "tss_imgs_sec_left.txt", img_ts_rel[:N])
        # intrinsics (fx fy cx cy)
        write_calib_fx_fy_cx_cy(out / "calib_undist_left.txt", P_rect_02)
        # imu csv (ns, gx gy gz ax ay az)
        write_imu_csv(out / "imu_data.csv", imu_ts_rel, oxts_packets)
        # gt camera poses (seconds-relative, t + quat(xyzw))
        write_gt_stamped_cam(out / "gt_stamped_left.txt", img_ts_rel[:N], T_w_c_for_img[:N])
        # yaml suggestion (units + Ti1c)
        write_yaml_hint(out / "prep_config_suggestion.yaml", Ti1c)

        # Sanity checks
        print(f"\n{'='*60}")
        print(f"PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Images written: {N}")
        print(f"Image timestamps: {len(img_ts_rel[:N])}")
        print(f"IMU measurements: {len(imu_ts_rel)}")
        print(f"IMU/Image ratio: {len(imu_ts_rel) / N:.2f} (should be >3.0)")
        print(f"Image time range: 0.0 - {img_ts_rel[N-1]:.2f}s")
        print(f"IMU time range: 0.0 - {imu_ts_rel[-1]:.2f}s")
        if imu_ts_rel[-1] < img_ts_rel[N-1]:
            print(f"⚠️  WARNING: IMU ends {img_ts_rel[N-1] - imu_ts_rel[-1]:.2f}s before last image!")
        else:
            print(f"✓ IMU coverage: OK (+{imu_ts_rel[-1] - img_ts_rel[N-1]:.2f}s beyond last image)")
        print(f"{'='*60}\n")

        if len(imu_ts_rel) >= 2:
            dt = float(np.mean(np.diff(imu_ts_rel)))
            print(f"[i] IMU mean dt ≈ {dt*1000:.2f} ms (~{1.0/dt:.1f} Hz)")
        print("[✓] Preprocessing complete.")

if __name__ == "__main__":
    main()

