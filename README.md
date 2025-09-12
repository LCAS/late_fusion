# Late Fusion Package

A ROS2 package for real-time late fusion of 2D and 3D object detections. This package implements the DeepFusion algorithm originally implemented in a ROS package by [prabuddhi02](https://github.com/Prabuddhi-05/deepfusion).

## Overview

This package implements a real-time late fusion algorithm that combines detections from both image and LiDAR sensors. The fusion process improves detection accuracy by leveraging the complementary strengths of both sensor modalities.

### Dependencies

The package relies on the following components:
- `image_detector_pkg` - Provides 2D image-based detections
- `lidar_detector_pkg` - Provides 3D LiDAR-based detections

### Nodes

The package provides two ROS2 nodes:

- **`late_fusion_node`** - Executes the fusion algorithm in real-time
- **`projection_node`** (optional) - Publishes real-time detection visualizations

## Features

- **Human Detection Focus**: Optimized for human detection in agricultural environments
- **Real-time Processing**: Designed for real-time robotic applications
- **Visual Feedback**: Optional image processing for detection visualization
- **Configurable**: Easy configuration through `config/late_fusion_config.yaml`

## Installation

### Option 1: Automated Installation (Experimental)

⚠️ **Warning**: This automated installation is experimental and may not work in all environments.

```bash
cd ros2_ws/src
git clone git@github.com:LCAS/late_fusion.git
cd late_fusion
bash install_all.sh
```

### Option 2: Manual Installation

1. Clone this repository:
```bash
cd ros2_ws/src
git clone git@github.com:LCAS/late_fusion.git
cd late_fusion
```

2. Install dependencies:
```bash
bash install_all.sh
```

3. Install the required detector packages:
   - **Image Detector**: [image_detector_pkg](https://github.com/ernstmv/image_detector_pkg)
   - **LiDAR Detector**: [3d_lidar_detector](https://github.com/ernstmv/lidar_detector_pkg/settings) 
Please follow the installation instructions for each detector package.

## Configuration

The package can be configured through the `config/late_fusion_config.yaml` file:

```yaml
late_fusion_node:
  ros__parameters:
    image_detections_topic: "yolo/detections"
    lidar_detections_topic: "lidar/detections"
    matching_topic: "deepfusion/matching"
    nonmatching_topic: "deepfusion/nonmatching"

projection_node:
  ros__parameters:
    input_matching_topic: "deepfusion/matching"
    input_nonmatching_topic: "deepfusion/nonmatching"
    input_image_topic: "camera/image_raw"
```

### Important Configuration Notes

- Ensure all topic names match your sensor setup
- Running both detectors simultaneously will reduce overall system performance
- For better performance, consider disabling projection nodes in detector packages if real-time visualization is not required

## Usage

### Quick Start

```bash
ros2 launch late_fusion_pkg late_fusion.launch.py
```

## Topics

### Subscribed Topics

- `yolo/detections` - 2D image detections
- `lidar/detections` - 3D LiDAR detections
- `camera/image_raw` - Raw camera images (for projection node)

### Published Topics

- `deepfusion/matching` - Fused detections with matches between sensors
- `deepfusion/nonmatching` - Detections that couldn't be matched between sensors

## Performance Considerations

- Running multiple detector nodes simultaneously will impact system performance
- Consider your computational resources when enabling all features
- Disable visualization nodes if real-time performance is critical

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

[Add your license information here]

## Acknowledgments

- Original DeepFusion algorithm by [Prabuddhi-05](https://github.com/Prabuddhi-05/deepfusion)
- LCAS (Lincoln Centre for Autonomous Systems)
