# Deep-Latent Space Based Data Assimilation Method for Integrating Multiple Body Tracking Techniques for Immersive XR Applications
This repository provides a motion dataset and pre-trained models designed for integrating multiple body tracking techniques in XR environments.  

## Dataset  
The dataset is intended for research in motion prediction, sensor fusion, and XR motion analysis. It enables training and evaluation of models that fuse data from multiple tracking systems, improving motion capture robustness and real-time accuracy. It combines Inside-Out Body Tracking (IOBT) and Markerless Motion Capture Data to improve real-time fullbody tracking accuracy. Additionally, the dataset includes multi-camera 3D pose estimation data as ground truth (GT).  
The dataset is available at [here](https://drive.google.com/uc?export=download&id=1cy4QFPCc_HHepIjy2ne-C3Yx8aii7hrW).  

### Data Types  
- **Inside-Out Body Tracking (IOBT)**: Utilizes the Meta Quest 3 VR headset's built-in cameras to track upper body movements, including hands and fingers, with high accuracy in real-time.  
- **Three-D Pose Tracker (TDPT)**: A markerless motion capture system using smartphone cameras that can track whole-body movements.  
- **Multi-Camera 3D Pose Estimation**: Provides ground truth (GT) data captured using a professional multi-camera setup for comprehensive and accurate reference measurements.  

### Categories  
- Daily – Everyday human activities
- Dance – Various dance movements
- Sport – Athletic and sports-related motions

### Example  
![Dataset Discrption](https://github.com/user-attachments/assets/a0dbd3fc-96b1-4e64-8fa9-327d22595124)

## Pre-trained Models
Pre-trained models provided in this repository include:

- [**Pose Auto-Encoder**](https://drive.google.com/uc?export=download&id=1IYUxbITgLdXOsQb1gcBUU3uXfRNwJKga) - Encodes and decodes pose data for compression and reconstruction  
- [**Time-Stepping Model**](https://drive.google.com/uc?export=download&id=1weXTcoywJ5ecRQEbhftT5JYkI2mZjT-y) – Distribution-based time stepping in latent space  
- [**D-LSAM (Deep Latent Space Assimilation Model)**](https://drive.google.com/uc?export=download&id=1aLZIyLW2Zx4IpCNdjyyJdza9NlxY3_FV) – Data assimilation in latent space  
