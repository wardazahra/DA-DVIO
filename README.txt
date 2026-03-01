# DA-DVIO Real-Time ROS Runner

Quick instructions for running DA-DVIO with live ROS data.

## Setup
```bash
# Activate environment
conda activate DEIO

# Check GPU
nvidia-smi  # Make sure no stuck processes
```

## Running

**Terminal 1 - Start ROS (Docker):**
```bash
docker start -i ros_noetic_network
source /opt/ros/noetic/setup.bash
roscore &
rosbag play /media/t2508/Tra/z11.bag --rate 1.0
```

**Terminal 2 - Run DEIO:**
```bash
cd /media/t2508/C477-48FA/DEIO
python realtime_slam_ros.py \
    --config /media/t2508/C477-48FA/DEIO/config/zed_imu.yaml \
    --network /home/t2508/warda/dpvo/DPVO-main/checkpoints/dpvo_depth_v3_000800.pth \
    --stride 2 \
    --save \
    --output saved_trajectories/output.txt
```

## Performance

- Processing: 60ms per frame
- Latency: 120ms (includes queue wait)
- Throughput: 7.5 Hz (stride=2)
- Frame drops: <1%

## Troubleshooting

**OOM Error:**
```bash
nvidia-smi  # Check for stuck processes
kill -9 [PID]  # Kill if needed
```


**Docker can't see bag:**
```bash
docker stop ros_noetic_network
docker rm ros_noetic_network
docker run -it --name ros_noetic_network --network host \
  -v /media/t2508:/media/t2508 \
  osrf/ros:noetic-desktop-full bash
```

## Notes

- Queue wait ~60ms is normal (previous frame processing)
- First 50-100 frames have higher latency (VI initialization)
- Depth matching tolerance: 150ms
