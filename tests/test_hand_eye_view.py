"""Check base-frame calibration arrow geometry without opening hardware."""

import numpy as np
from scipy.spatial.transform import Rotation

from so101_control.calibration import measured_gripper_T_tag
from so101_control.hand_eye_view import HandEyeView, _pose_arrows


def test_ee_and_tag_arrows_use_fk_base_coordinates():
    class Arrows:
        def __init__(self, **kwargs):
            self.origins = np.asarray(kwargs["origins"])
            self.vectors = np.asarray(kwargs["vectors"])
            self.labels = kwargs["labels"]

    class Rerun:
        Arrows3D = Arrows

    base_T_gripper = np.eye(4)
    base_T_gripper[:3, :3] = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    base_T_gripper[:3, 3] = [0.23, 0.04, 0.15]
    tag = base_T_gripper @ measured_gripper_T_tag()
    visual = _pose_arrows(Rerun, tag, "tag")
    assert np.allclose(visual.origins, [tag[:3, 3]] * 3)
    assert np.allclose(visual.vectors, tag[:3, :3].T * 0.055)
    assert visual.labels == ["tag X", "tag Y", "tag Z"]


def test_each_camera_candidate_remains_visible_after_confirmation():
    class Recording:
        def __init__(self):
            self.calls = []

        def log(self, path, *objects, **kwargs):
            self.calls.append((path, objects))

        def Arrows3D(self, **kwargs):
            return kwargs

        def CoordinateFrame(self, name):
            return name

        def Transform3D(self, **kwargs):
            return kwargs

        def TextLog(self, text):
            return text

    rr = Recording()
    view = HandEyeView.__new__(HandEyeView)
    view.rr, view.root = rr, "/robot"
    camera = np.eye(4)
    view.show_camera_estimate(camera, sample_count=1)
    view.show_camera_estimate(camera, sample_count=2)
    view.show_camera_estimate(camera, sample_count=2, confirmed=True)
    paths = [path for path, _ in rr.calls]
    assert "/robot/calibration_axes/candidates/1" in paths
    assert paths.count("/robot/calibration_axes/candidates/2") == 2
    assert rr.calls[-1] == ("calibration/camera_status",
                            ("ACCEPTED: 2 poses; verify multiple viewpoints before trusting camera extrinsics",))