"""Convert CSV motion data to GMR format .pkl for use with the retarget pipeline.

This script converts CSV files (from MuJoCo/simulator recording) to GMR format .pkl,
which can then be processed by single_retarget.py or dataset_retarget.py to generate
the final Legged Lab .pkl with key_body_pos.

CSV expected columns:
    time_sec, root_pos_x, root_pos_y, root_pos_z,
    root_quat_w, root_quat_x, root_quat_y, root_quat_z,
    base_lin_vel_x, base_lin_vel_y, base_lin_vel_z,
    base_ang_vel_x, base_ang_vel_y, base_ang_vel_z,
    FR_hip_joint, FR_thigh_joint, FR_calf_joint,
    FL_hip_joint, FL_thigh_joint, FL_calf_joint,
    RR_hip_joint, RR_thigh_joint, RR_calf_joint,
    RL_hip_joint, RL_thigh_joint, RL_calf_joint,
    <joint velocities - ignored>

GMR output format:
    fps: int
    root_pos: np.array (N, 3)
    root_rot: np.array (N, 4)  - quaternion (x, y, z, w)
    dof_pos: np.array (N, 12) - joint positions in GMR order (FL, FR, RL, RR)
                              - MuJoCo (CSV) sign convention is preserved as-is;
                              the sign flip for right-side thigh/calf joints is
                              applied later in gmr_to_lab.py when converting to
                              Isaac Lab (PhysX) convention.
    local_body_pos: None
    link_body_list: None

Usage:
    python scripts/tools/retarget/csv_to_gmr.py \
        --input_csv TB.csv \
        --output_pkl output/gmr_walk.pkl \
        [--fps 500] \
        [--frame_range START END] \
        [--loop {wrap,clamp}]

    python scripts/tools/retarget/csv_to_gmr.py \
        --input_dir . \
        --output_dir temp/gmr_data \
        [--fps 500] \
        [--frame_range START END] \
        [--loop {wrap,clamp}]
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


# CSV column indices for joint positions (0-based)
# CSV order: FR, FL, RR, RL
# GMR order (must match atdog2.yaml gmr_dof_names): FL, FR, RL, RR
# Mapping: GMR index i -> CSV column index
GMR_DOF_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV motion data to GMR .pkl")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--output_pkl", type=str, help="Path to output GMR .pkl file")
    parser.add_argument("--input_dir", type=str, help="Directory containing input CSV files")
    parser.add_argument("--output_dir", type=str, help="Directory for output GMR .pkl files")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override fps (default: auto-detect from time_sec column)",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Frame range to extract (0-based index). Example: --frame_range 0 1000",
    )
    parser.add_argument(
        "--loop",
        type=str,
        choices=["wrap", "clamp"],
        default="clamp",
        help="Loop mode for the motion (default: clamp)",
    )
    args = parser.parse_args()

    has_single = args.input_csv is not None or args.output_pkl is not None
    has_batch = args.input_dir is not None or args.output_dir is not None

    if has_single and has_batch:
        raise ValueError("请使用单文件模式(--input_csv/--output_pkl)或批量模式(--input_dir/--output_dir)，不要混用。")

    if has_single:
        if args.input_csv is None or args.output_pkl is None:
            raise ValueError("单文件模式下必须同时提供 --input_csv 和 --output_pkl。")
    elif has_batch:
        if args.input_dir is None or args.output_dir is None:
            raise ValueError("批量模式下必须同时提供 --input_dir 和 --output_dir。")
    else:
        raise ValueError("请提供单文件模式(--input_csv/--output_pkl)或批量模式(--input_dir/--output_dir)参数。")

    return args


def load_csv_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    return rows


def build_gmr_data(rows: list[dict[str, str]], fps_override: int | None, frame_range: list[int] | None):
    num_rows = len(rows)
    root_pos = np.zeros((num_rows, 3), dtype=np.float64)
    root_rot_xyzw = np.zeros((num_rows, 4), dtype=np.float64)

    for i, row in enumerate(rows):
        root_pos[i, 0] = float(row["root_pos_x"])
        root_pos[i, 1] = float(row["root_pos_y"])
        root_pos[i, 2] = float(row["root_pos_z"])

        qw = float(row["root_quat_w"])
        qx = float(row["root_quat_x"])
        qy = float(row["root_quat_y"])
        qz = float(row["root_quat_z"])
        root_rot_xyzw[i] = [qx, qy, qz, qw]

    dof_pos = np.zeros((num_rows, len(GMR_DOF_ORDER)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, dof_name in enumerate(GMR_DOF_ORDER):
            dof_pos[i, j] = float(row[dof_name])

    if fps_override is not None:
        fps = fps_override
    else:
        if num_rows < 2:
            raise ValueError("CSV 至少需要 2 帧才能自动推断 fps；请使用 --fps 手动指定。")
        t0 = float(rows[0]["time_sec"])
        t1 = float(rows[1]["time_sec"])
        dt = t1 - t0
        fps = int(round(1.0 / dt))

    start_frame = 0
    end_frame = num_rows
    if frame_range:
        start_frame = frame_range[0]
        end_frame = min(frame_range[1], num_rows)

    root_pos = root_pos[start_frame:end_frame]
    root_rot_xyzw = root_rot_xyzw[start_frame:end_frame]
    dof_pos = dof_pos[start_frame:end_frame]

    gmr_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    return gmr_data, start_frame, end_frame


def save_gmr_data(gmr_data: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(gmr_data, f)


def convert_one(csv_path: Path, out_path: Path, fps_override: int | None, frame_range: list[int] | None):
    rows = load_csv_rows(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    gmr_data, start_frame, end_frame = build_gmr_data(rows, fps_override, frame_range)
    save_gmr_data(gmr_data, out_path)

    actual_frames = gmr_data["root_pos"].shape[0]
    fps = gmr_data["fps"]
    dt = None
    if fps_override is None and len(rows) >= 2:
        dt = float(rows[1]["time_sec"]) - float(rows[0]["time_sec"])

    print(f"Using frames [{start_frame}, {end_frame}) -> {actual_frames} frames")
    if dt is not None:
        print(f"Auto-detected fps={fps} (dt={dt:.6f}s)")
    print(f"Saved GMR .pkl to: {out_path}")
    print(f"  fps:      {fps}")
    print(f"  frames:   {actual_frames}")
    print(f"  duration: {actual_frames / fps:.2f}s")
    print(f"  root_pos: {gmr_data['root_pos'].shape}")
    print(f"  root_rot: {gmr_data['root_rot'].shape} (x,y,z,w)")
    print(f"  dof_pos:  {gmr_data['dof_pos'].shape}")


def convert_batch(input_dir: Path, output_dir: Path, fps_override: int | None, frame_range: list[int] | None):
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"在目录中未找到 CSV 文件: {input_dir}")

    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    for csv_path in csv_files:
        out_path = output_dir / f"{csv_path.stem}.pkl"
        print(f"\n=== Converting {csv_path.name} ===")
        convert_one(csv_path, out_path, fps_override, frame_range)

    print(f"\nBatch conversion complete. Output dir: {output_dir}")


def main():
    args = parse_args()

    if args.input_csv is not None:
        out_path = Path(args.output_pkl)
        convert_one(Path(args.input_csv), out_path, args.fps, args.frame_range)
        print("\nNext step: use single_retarget.py to convert to Legged Lab format:")
        print("  python scripts/tools/retarget/single_retarget.py \\")
        print("      --robot atdog2 \\")
        print(f"      --input_file {out_path} \\")
        print("      --output_file <output_path>/walk.pkl \\")
        print("      --config_file scripts/tools/retarget/config/atdog2.yaml \\")
        print(f"      --loop {args.loop} --headless")
    else:
        convert_batch(Path(args.input_dir), Path(args.output_dir), args.fps, args.frame_range)


if __name__ == "__main__":
    main()
