# Sources and attribution

- Kinematics is computed in this repository from the URDF. The servo bus uses [feetech-servo-sdk](https://pypi.org/project/feetech-servo-sdk/) (the Waveshare/Feetech STS protocol library).
- [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), Apache-2.0, commit `a758567c3978dfeefe282ede0500085a48fe8f78`. Setup downloads their original SO-101 follower URDF and meshes, plus `LICENSE.upstream`, into `assets/SO101/`. The URDF is generated from Onshape CAD via `onshape-to-robot` according to upstream. Keep that license with any redistributed assets.
