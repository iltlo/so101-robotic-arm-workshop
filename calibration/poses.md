# Hand-eye auto poses

Tool poses in the base frame. Roll stays at −170° so the wrist stays off the stop.
The second block is the first mirrored through the base XZ plane (`y → −y`, `yaw → −yaw`).

```text
# recorded side
0.24  0.03  0.13  -170   3    8
0.23  0.03  0.20  -170   2    8
0.18 -0.15  0.20  -170   7  -48
0.21 -0.16  0.11  -170  -6  -44
0.31 -0.14  0.09  -170  -2  -28

# mirrored across base X (XZ plane)
0.24 -0.03  0.13  -170   3   -8
0.23 -0.03  0.20  -170   2   -8
0.18  0.15  0.20  -170   7   48
0.21  0.16  0.11  -170  -6   44
0.31  0.14  0.09  -170  -2   28
```

Run:

```sh
uv run python calibration/auto_hand_eye.py --live
```
