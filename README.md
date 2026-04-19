# autonomous_car_pkg

ROS2 package for autonomous car navigation using the Edubot.
Implements lane following, SLAM mapping, obstacle avoidance and basic sign recognition.

---

## Package structure

```
autonomous_car_pkg/
├── autonomous_car_pkg/
│   ├── lane_detection_node.py   # vision: white/yellow/orange lines
│   ├── navigation_node.py       # PD controller → /cmd_vel
│   ├── obstacle_avoidance.py    # LiDAR stop/go logic
│   ├── behavior_controller.py   # top-level state machine
│   └── lane_map_node.py         # accumulates lane trail in /map frame
├── launch/
│   └── full_system.launch.py
├── config/
│   └── slam_toolbox.yaml
├── package.xml
└── setup.py
```

---

## Node graph (topic connections)

```
/camera/bottom/image_raw ──► lane_detection_node ──► /lane/error
                                                  ──► /lane/end_of_road
                                                  ──► /lane/markers
                                                  ──► /lane/debug_image

/lane/error        ──► navigation_node ──► /cmd_vel
/lane/end_of_road  ──►    │
/behavior/state    ──►    │

/scan ──► obstacle_avoidance ──► /behavior/state
                             ──► /obstacles/markers

/sign/detected ──► behavior_controller ──► /behavior/state

/lane/markers ──► lane_map_node ──► /map/lane_markers  (RViz)
                              ──► /map/lane_path       (RViz)

/scan + /odom ──► slam_toolbox ──► /map  (OccupancyGrid for RViz)
```

---

## Installation

```bash
# 1. Create a separate workspace (required by project spec)
mkdir -p ~/car_ws/src && cd ~/car_ws/src
git clone <your-repo-url> autonomous_car_pkg

# 2. Install slam_toolbox
sudo apt install ros-humble-slam-toolbox

# 3. Build
cd ~/car_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Running

```bash
# Full system (robot laptop)
ros2 launch autonomous_car_pkg full_system.launch.py

# With RViz on remote machine (same ROS_DOMAIN_ID)
ros2 launch autonomous_car_pkg full_system.launch.py use_rviz:=true
```

### RViz displays to add manually (first time):
- **Map** → topic `/map`
- **MarkerArray** → topic `/map/lane_markers`  (white line trail)
- **MarkerArray** → topic `/obstacles/markers` (cones / obstacles)
- **Path** → topic `/map/lane_path`
- **Image** → topic `/lane/debug_image`        (tuning camera view)

---

## Tuning lane detection

The HSV thresholds are ROS parameters — tune while running:

```bash
# Check current white line detection
ros2 run rqt_image_view rqt_image_view /lane/debug_image

# Adjust thresholds live (no restart needed)
ros2 param set /lane_detection_node white_v_min 140   # less strict white
ros2 param set /lane_detection_node white_s_max 80    # allow slightly coloured white

# Adjust PD gains live
ros2 param set /navigation_node kp 1.0
ros2 param set /navigation_node kd 0.2

# Slow down (useful for first test)
ros2 param set /navigation_node base_speed 0.10
```

### What the debug image shows:
- **Blue tint**   = detected white pixels
- **Yellow tint** = detected yellow pixels
- **Orange tint** = detected orange (end-of-road) pixels
- **Green line**  = target white line position
- **Cyan line**   = actual detected white line centroid

---

## Tuning the controller

| Parameter | Default | Effect |
|-----------|---------|--------|
| `kp`      | 0.8     | Higher = faster correction, but oscillates |
| `kd`      | 0.15    | Higher = smoother, but sluggish |
| `base_speed` | 0.15 m/s | Faster = harder to control |
| `stop_distance` | 0.40 m | Closer = less sensitive to false positives |

**First test checklist:**
1. Start with `base_speed: 0.08` (very slow).
2. Watch `/lane/debug_image` — confirm white line is detected.
3. Check `/lane/error` with `ros2 topic echo /lane/error`:
   - should be near 0 when centred.
   - positive when robot drifts left (white line too close on right).
4. Let robot drive straight, then increase speed gradually.

---

## Known issues and workarounds

**White line not detected indoors under fluorescent light:**
- Lower `white_v_min` to 140 or even 120.
- Tape may appear slightly blue/yellow under LED — adjust `white_s_max` up to 100.

**Robot oscillates side to side:**
- Reduce `kp` or increase `kd`.
- Check if `/lane/error` is noisy (jittery detections) — add blur to camera image.

**SLAM map drifts:**
- Make sure `/odom` is publishing at ≥ 20 Hz.
- Check that `/scan` topic name matches your robot's LiDAR topic.

**TF tree not connected:**
- `ros2 run tf2_tools view_frames` — check that map → odom → base_link → laser is complete.
- slam_toolbox publishes `map → odom`; your robot driver must publish `odom → base_link`.

---

## Sign detection (extra credit)

The `/sign/detected` topic accepts `String` messages with values:
- `"stop"` → 5 s pause
- `"road_closed"` → immediate turn-around
- `"one_way"` → stop (don't traverse wrong way)

To add sign recognition, create `sign_detection_node.py` that:
1. Subscribes to `/camera/front/image_raw`.
2. Uses a colour-based detector or a small YOLO model.
3. Publishes a `std_msgs/String` to `/sign/detected`.

The behavior_controller is already wired to receive these.
