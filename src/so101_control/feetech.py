"""STS3215 bus for the SO-101 follower. A full count is 360 degrees.

Present_Position is already Actual_Position minus the homing offset stored in
the servo. Arm joints convert that count with the midpoint of the calibrated
min and max. The gripper stays 0 closed to 100 open.
"""

from pathlib import Path
import json
import threading
import time

# One turn of an STS3215, and the largest count (resolution - 1).
RESOLUTION = 4096
MAX_TICK = RESOLUTION - 1
MODEL_NUMBER = 777
BAUD_RATE = 1_000_000

# Wrist roll is a full turn. Range capture does not sweep it.
FULL_TURN = "wrist_roll"
# name, id, "degrees" or "percent"
MOTORS = (
    ("shoulder_pan", 1, "degrees"),
    ("shoulder_lift", 2, "degrees"),
    ("elbow_flex", 3, "degrees"),
    ("wrist_flex", 4, "degrees"),
    ("wrist_roll", 5, "degrees"),
    ("gripper", 6, "percent"),
)

# STS/SMS register: address, byte length, sign-magnitude bit or None.
# Addresses follow the Feetech SMS/STS control table.
_REGS = {
    "Model_Number": (3, 2, None),
    "Return_Delay_Time": (7, 1, None),
    "Min_Position_Limit": (9, 2, None),
    "Max_Position_Limit": (11, 2, None),
    "Phase": (18, 1, None),
    "Homing_Offset": (31, 2, 11),
    "Torque_Enable": (40, 1, None),
    "Acceleration": (41, 1, None),
    "Goal_Position": (42, 2, 15),
    "Lock": (55, 1, None),
    "Present_Position": (56, 2, 15),
    "Maximum_Acceleration": (85, 1, None),
}


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    """Feetech sign-magnitude: the sign lives at `sign_bit`, the rest is the magnitude."""
    value = int(value)
    limit = (1 << sign_bit) - 1
    if abs(value) > limit:
        raise ValueError(f"Value {value} does not fit in sign-magnitude bit {sign_bit}")
    if value < 0:
        return abs(value) | (1 << sign_bit)
    return value


def decode_sign_magnitude(encoded: int, sign_bit: int) -> int:
    encoded = int(encoded)
    magnitude = encoded & ((1 << sign_bit) - 1)
    if encoded & (1 << sign_bit):
        return -magnitude
    return magnitude


def ticks_to_degrees(ticks: float, range_min: float, range_max: float) -> float:
    """Zero is the midpoint of the calibrated range. A full count is 360 degrees."""
    if range_max == range_min:
        raise ValueError("Motor range is empty")
    mid = (float(range_min) + float(range_max)) / 2.0
    return (float(ticks) - mid) * 360.0 / MAX_TICK


def degrees_to_ticks(degrees: float, range_min: float, range_max: float) -> int:
    if range_max == range_min:
        raise ValueError("Motor range is empty")
    mid = (float(range_min) + float(range_max)) / 2.0
    return int((float(degrees) * MAX_TICK / 360.0) + mid)


def ticks_to_percent(ticks: float, range_min: float, range_max: float, drive_mode: int = 0) -> float:
    """0 at range_min and 100 at range_max, unless drive_mode flips the jaw."""
    if range_max == range_min:
        raise ValueError("Motor range is empty")
    bounded = min(float(range_max), max(float(range_min), float(ticks)))
    norm = ((bounded - float(range_min)) / (float(range_max) - float(range_min))) * 100.0
    return 100.0 - norm if drive_mode else norm


def percent_to_ticks(percent: float, range_min: float, range_max: float, drive_mode: int = 0) -> int:
    if range_max == range_min:
        raise ValueError("Motor range is empty")
    value = float(percent)
    if drive_mode:
        value = 100.0 - value
    value = min(100.0, max(0.0, value))
    return int((value / 100.0) * (float(range_max) - float(range_min)) + float(range_min))


def _factory(cal: dict) -> bool:
    return cal["range_min"] == 0 and cal["range_max"] == MAX_TICK


class MemoryLink:
    """Register store for tests. Values are the on-wire integers."""

    def __init__(self):
        self.is_open = False
        self.values: dict[tuple[int, int], int] = {}

    def open_port(self) -> None:
        self.is_open = True

    def close_port(self) -> None:
        self.is_open = False

    def read(self, motor_id: int, address: int, length: int) -> int:
        if not self.is_open:
            raise RuntimeError("Serial port is closed")
        return int(self.values.get((motor_id, address), 0))

    def write(self, motor_id: int, address: int, length: int, encoded: int) -> None:
        if not self.is_open:
            raise RuntimeError("Serial port is closed")
        self.values[(motor_id, address)] = int(encoded)

    def sync_read(self, address: int, length: int, motor_ids) -> dict[int, int]:
        return {motor_id: self.read(motor_id, address, length) for motor_id in motor_ids}

    def sync_write(self, address: int, length: int, values: dict[int, int]) -> None:
        for motor_id, encoded in values.items():
            self.write(motor_id, address, length, encoded)


def _patch_packet_timeout(port_handler) -> None:
    """The PyPI Feetech SDK times out too soon. Give each packet the extra 50 ms it needs."""

    def set_packet_timeout(packet_length):
        port_handler.packet_start_time = port_handler.getCurrentTime()
        port_handler.packet_timeout = (
            (port_handler.tx_time_per_byte * packet_length)
            + (port_handler.tx_time_per_byte * 3.0)
            + 50
        )

    port_handler.setPacketTimeout = set_packet_timeout


class SdkLink:
    """scservo_sdk port. Imported only when a real adapter is opened."""

    def __init__(self, port: str):
        import scservo_sdk as scs

        self._scs = scs
        self.port_handler = scs.PortHandler(str(port))
        _patch_packet_timeout(self.port_handler)
        self.packet = scs.PacketHandler(0)
        self._ok = scs.COMM_SUCCESS

    def open_port(self) -> None:
        if not self.port_handler.openPort():
            raise RuntimeError(f"Could not open {self.port_handler.port_name}")
        if not self.port_handler.setBaudRate(BAUD_RATE):
            self.port_handler.closePort()
            raise RuntimeError(f"Could not set {BAUD_RATE} baud on {self.port_handler.port_name}")

    def close_port(self) -> None:
        self.port_handler.closePort()

    def _check(self, result, error) -> None:
        if result != self._ok:
            raise RuntimeError(self.packet.getTxRxResult(result))
        if error:
            raise RuntimeError(self.packet.getRxPacketError(error))

    def _call(self, fn):
        try:
            return fn()
        except OSError as exc:
            port = getattr(self.port_handler, "port_name", "the serial port")
            raise RuntimeError(f"Serial port {port} failed: {exc}") from exc

    def read(self, motor_id: int, address: int, length: int) -> int:
        def run():
            if length == 1:
                value, result, error = self.packet.read1ByteTxRx(self.port_handler, motor_id, address)
            elif length == 2:
                value, result, error = self.packet.read2ByteTxRx(self.port_handler, motor_id, address)
            else:
                raise ValueError(f"Unsupported register length {length}")
            self._check(result, error)
            return int(value)

        return self._call(run)

    def write(self, motor_id: int, address: int, length: int, encoded: int) -> None:
        def run():
            value = int(encoded)
            if length == 1:
                result, error = self.packet.write1ByteTxRx(self.port_handler, motor_id, address, value)
            elif length == 2:
                result, error = self.packet.write2ByteTxRx(self.port_handler, motor_id, address, value)
            else:
                raise ValueError(f"Unsupported register length {length}")
            self._check(result, error)

        self._call(run)

    def sync_read(self, address: int, length: int, motor_ids) -> dict[int, int]:
        def run():
            reader = self._scs.GroupSyncRead(self.port_handler, self.packet, address, length)
            for motor_id in motor_ids:
                if not reader.addParam(motor_id):
                    raise RuntimeError(f"Could not add id {motor_id} to the sync read")
            result = reader.txRxPacket()
            if result != self._ok:
                raise RuntimeError(self.packet.getTxRxResult(result))
            values = {}
            for motor_id in motor_ids:
                if not reader.isAvailable(motor_id, address, length):
                    raise RuntimeError(f"No sync read for servo id {motor_id}")
                values[motor_id] = int(reader.getData(motor_id, address, length))
            return values

        return self._call(run)

    def sync_write(self, address: int, length: int, values: dict[int, int]) -> None:
        def run():
            writer = self._scs.GroupSyncWrite(self.port_handler, self.packet, address, length)
            for motor_id, encoded in values.items():
                encoded = int(encoded)
                if length == 1:
                    data = [encoded & 0xFF]
                else:
                    data = [encoded & 0xFF, (encoded >> 8) & 0xFF]
                if not writer.addParam(motor_id, data):
                    raise RuntimeError(f"Could not add id {motor_id} to the sync write")
            result = writer.txPacket()
            if result != self._ok:
                raise RuntimeError(self.packet.getTxRxResult(result))
            writer.clearParam()

        self._call(run)


class FeetechBus:
    """One lock around each serial transaction. Motion and e-stop share it."""

    def __init__(self, port: str, link=None, *, calibration_path: Path | None = None):
        self.port = str(port)
        self.link = link
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.calibration: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._open = False
        self._ids = {name: motor_id for name, motor_id, _mode in MOTORS}
        self._modes = {name: mode for name, _motor_id, mode in MOTORS}
        self._range_stop = threading.Event()
        self._range_thread = None

    @property
    def is_connected(self) -> bool:
        return self._open

    @property
    def is_calibrated(self) -> bool:
        """True when a saved range exists. Wrist roll may still use the full turn."""
        if set(self.calibration) != set(self._ids):
            return False
        return not all(_factory(cal) for cal in self.calibration.values())

    def connect(self) -> None:
        if self.link is None:
            self.link = SdkLink(self.port)
        self.link.open_port()
        self._open = True
        try:
            self._check_models()
            self._prepare_servos()
            self._load_calibration(apply=True)
            if self.is_calibrated:
                self.enable_torque()
        except BaseException:
            self.disconnect(False)
            raise

    def disconnect(self, disable_torque: bool = True) -> None:
        self._range_stop.set()
        thread = self._range_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        with self._lock:
            if not self._open:
                return
            if disable_torque:
                try:
                    self._torque_locked(False)
                except Exception:
                    pass
            try:
                if self.link is not None:
                    self.link.close_port()
            finally:
                self._open = False

    def sync_read(self, name: str, motor_names=None) -> dict[str, float]:
        if name != "Present_Position":
            raise ValueError(f"Unsupported sync read {name}")
        names = list(motor_names) if motor_names is not None else [item[0] for item in MOTORS]
        raw = self._read_names(name, names)
        return {motor: self._to_user(motor, raw[motor]) for motor in names}

    def sync_write(self, name: str, values: dict[str, float]) -> None:
        if name != "Goal_Position":
            raise ValueError(f"Unsupported sync write {name}")
        if not values:
            return
        encoded = {}
        for motor, user_value in values.items():
            if motor not in self._ids:
                raise ValueError(f"Unknown joint {motor}")
            ticks = self._from_user(motor, float(user_value))
            encoded[self._ids[motor]] = self._encode("Goal_Position", ticks)
        address, length, _sign = _REGS["Goal_Position"]
        with self._lock:
            self._require_open()
            self.link.sync_write(address, length, encoded)

    def disable_torque(self, *_args, **_kwargs) -> None:
        with self._lock:
            self._require_open()
            self._torque_locked(False)

    def enable_torque(self, *_args, **_kwargs) -> None:
        with self._lock:
            self._require_open()
            self._torque_locked(True)

    def set_midpoint(self) -> dict[str, int]:
        """Torque off, then set the homing offset so the current pose reads as half a turn."""
        self.disable_torque()
        for name in self._ids:
            self._write_user_raw(name, "Homing_Offset", 0)
            self._write_raw(name, "Min_Position_Limit", 0)
            self._write_raw(name, "Max_Position_Limit", MAX_TICK)
        time.sleep(0.05)
        present = self._read_names("Present_Position", list(self._ids))
        offsets = {}
        for name, ticks in present.items():
            offset = int(ticks) - (MAX_TICK // 2)
            self._write_user_raw(name, "Homing_Offset", offset)
            offsets[name] = offset
        self._update_stored_homing(offsets)
        self._load_calibration()
        return offsets

    def start_range_capture(self) -> None:
        """Remember min and max present ticks while the operator moves the arm."""
        self.disable_torque()
        self._range_stop.set()
        thread = self._range_thread
        if thread is not None:
            thread.join(timeout=1)
        present = self._read_names("Present_Position", list(self._ids))
        self._mins = dict(present)
        self._maxs = dict(present)
        self._range_stop = threading.Event()

        def watch() -> None:
            while not self._range_stop.is_set():
                try:
                    sample = self._read_names("Present_Position", list(self._ids))
                except Exception:
                    if self._range_stop.is_set():
                        return
                    time.sleep(0.05)
                    continue
                for name, ticks in sample.items():
                    self._mins[name] = min(self._mins[name], ticks)
                    self._maxs[name] = max(self._maxs[name], ticks)
                time.sleep(0.05)

        self._range_thread = threading.Thread(target=watch, name="feetech-range", daemon=True)
        self._range_thread.start()

    def stop_range_capture(self) -> dict[str, dict]:
        if self._range_thread is None:
            raise RuntimeError("Range capture is not running")
        self._range_stop.set()
        self._range_thread.join(timeout=1)
        self._range_thread = None
        saved = {}
        for name in self._ids:
            if name == FULL_TURN:
                low, high = 0, MAX_TICK
            else:
                low = int(self._mins[name])
                high = int(self._maxs[name])
                if high <= low:
                    raise ValueError(f"{name} did not move during range capture")
            self._write_raw(name, "Min_Position_Limit", low)
            self._write_raw(name, "Max_Position_Limit", high)
            saved[name] = {
                "id": self._ids[name],
                "drive_mode": 0,
                "homing_offset": self.calibration.get(name, {}).get("homing_offset", 0),
                "range_min": low,
                "range_max": high,
            }
        self.calibration = saved
        self._write_calibration_file(saved)
        return saved

    def range_snapshot(self) -> dict[str, dict] | None:
        if self._range_thread is None or not self._range_thread.is_alive():
            return None
        return {
            name: {"min": self._mins[name], "max": self._maxs[name]}
            for name in self._ids
        }

    def _check_models(self) -> None:
        for name, motor_id in self._ids.items():
            model = self._read_raw(name, "Model_Number")
            if model != MODEL_NUMBER:
                raise RuntimeError(
                    f"Servo id {motor_id} ({name}) reported model {model}, expected {MODEL_NUMBER}"
                )

    def _prepare_servos(self) -> None:
        """Short return delay and position tracking, with angle feedback in 0..4095."""
        self.disable_torque()
        for name in self._ids:
            self._write_raw(name, "Return_Delay_Time", 0)
            self._write_raw(name, "Acceleration", 254)
            self._write_raw(name, "Maximum_Acceleration", 254)
            phase = self._read_raw(name, "Phase")
            if phase & 0x10:
                self._write_raw(name, "Phase", phase & ~0x10)

    def _load_calibration(self, *, apply: bool = False) -> None:
        """Use config/motors.json when it lists every joint. Otherwise read the servos.

        `apply` writes the file onto the servos when their homing offset or limits differ.
        A full-turn wrist is a normal saved range, not a missing calibration.
        """
        stored = self._read_stored_calibration()
        if stored is not None:
            if apply:
                self._write_calibration(stored)
            self.calibration = stored
            return
        loaded = {}
        for name in self._ids:
            loaded[name] = {
                "id": self._ids[name],
                "drive_mode": 0,
                "homing_offset": self._read_decoded(name, "Homing_Offset"),
                "range_min": self._read_raw(name, "Min_Position_Limit"),
                "range_max": self._read_raw(name, "Max_Position_Limit"),
            }
        self.calibration = loaded

    def _read_stored_calibration(self) -> dict | None:
        path = self.calibration_path
        if path is None or not path.is_file():
            return None
        stored = json.loads(path.read_text())
        if not isinstance(stored, dict):
            raise ValueError(f"{path.name} must be a mapping")
        loaded = {}
        for name in self._ids:
            cal = stored.get(name)
            if not isinstance(cal, dict):
                raise ValueError(f"{path.name} is missing {name}")
            try:
                low = int(cal["range_min"])
                high = int(cal["range_max"])
                offset = int(cal.get("homing_offset") or 0)
                drive = int(cal.get("drive_mode") or 0)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid calibration for {name} in {path.name}") from exc
            if high <= low:
                raise ValueError(f"Invalid range for {name} in {path.name}")
            loaded[name] = {
                "id": self._ids[name],
                "drive_mode": drive,
                "homing_offset": offset,
                "range_min": low,
                "range_max": high,
            }
        return loaded

    def _write_calibration(self, stored: dict) -> None:
        """Copy a saved calibration into the servos. Skip registers that already match."""
        for name, cal in stored.items():
            if self._read_decoded(name, "Homing_Offset") != cal["homing_offset"]:
                self._write_user_raw(name, "Homing_Offset", cal["homing_offset"])
            if self._read_raw(name, "Min_Position_Limit") != cal["range_min"]:
                self._write_raw(name, "Min_Position_Limit", cal["range_min"])
            if self._read_raw(name, "Max_Position_Limit") != cal["range_max"]:
                self._write_raw(name, "Max_Position_Limit", cal["range_max"])

    def _update_stored_homing(self, offsets: dict[str, int]) -> None:
        path = self.calibration_path
        if path is None or not path.is_file():
            return
        stored = json.loads(path.read_text())
        if not isinstance(stored, dict):
            return
        for name, offset in offsets.items():
            cal = stored.get(name)
            if isinstance(cal, dict):
                cal["homing_offset"] = int(offset)
        self._write_calibration_file(stored)

    def _write_calibration_file(self, saved: dict) -> None:
        if self.calibration_path is None:
            return
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_path.write_text(json.dumps(saved, indent=2) + "\n")

    def _torque_locked(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        for name in self._ids:
            self._write_raw_locked(name, "Torque_Enable", value)
            self._write_raw_locked(name, "Lock", value)

    def _to_user(self, name: str, ticks: int) -> float:
        cal = self._cal(name)
        if self._modes[name] == "degrees":
            return ticks_to_degrees(ticks, cal["range_min"], cal["range_max"])
        return ticks_to_percent(ticks, cal["range_min"], cal["range_max"], cal["drive_mode"])

    def _from_user(self, name: str, value: float) -> int:
        cal = self._cal(name)
        if self._modes[name] == "degrees":
            return degrees_to_ticks(value, cal["range_min"], cal["range_max"])
        return percent_to_ticks(value, cal["range_min"], cal["range_max"], cal["drive_mode"])

    def _cal(self, name: str) -> dict:
        try:
            return self.calibration[name]
        except KeyError as exc:
            raise RuntimeError(f"No calibration loaded for {name}") from exc

    def _encode(self, register: str, value: int) -> int:
        _address, _length, sign = _REGS[register]
        if sign is None:
            return int(value)
        return encode_sign_magnitude(int(value), sign)

    def _decode(self, register: str, encoded: int) -> int:
        _address, _length, sign = _REGS[register]
        if sign is None:
            return int(encoded)
        return decode_sign_magnitude(int(encoded), sign)

    def _read_names(self, register: str, names) -> dict[str, int]:
        address, length, _sign = _REGS[register]
        ids = [self._ids[name] for name in names]
        with self._lock:
            self._require_open()
            raw = self.link.sync_read(address, length, ids)
        by_id = {motor_id: self._decode(register, value) for motor_id, value in raw.items()}
        return {name: by_id[self._ids[name]] for name in names}

    def _read_raw(self, name: str, register: str) -> int:
        address, length, _sign = _REGS[register]
        with self._lock:
            self._require_open()
            return int(self.link.read(self._ids[name], address, length))

    def _read_decoded(self, name: str, register: str) -> int:
        return self._decode(register, self._read_raw(name, register))

    def _write_raw(self, name: str, register: str, value: int) -> None:
        with self._lock:
            self._require_open()
            self._write_raw_locked(name, register, value)

    def _write_user_raw(self, name: str, register: str, value: int) -> None:
        self._write_raw(name, register, int(value))

    def _write_raw_locked(self, name: str, register: str, value: int) -> None:
        address, length, sign = _REGS[register]
        encoded = encode_sign_magnitude(int(value), sign) if sign is not None else int(value)
        self.link.write(self._ids[name], address, length, encoded)

    def _require_open(self) -> None:
        if not self._open or self.link is None:
            raise RuntimeError("Feetech bus is closed")


class FeetechFollower:
    """get_observation / send_action surface used by SO101."""

    def __init__(self, port: str, link=None, *, calibration_path: Path | None = None):
        self.bus = FeetechBus(port, link=link, calibration_path=calibration_path)

    def connect(self) -> None:
        self.bus.connect()

    def disconnect(self, disable_torque: bool = True) -> None:
        self.bus.disconnect(disable_torque)

    def close(self) -> None:
        self.disconnect(False)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def get_observation(self) -> dict:
        if not self.bus.is_connected:
            raise RuntimeError("Feetech arm is closed")
        positions = self.bus.sync_read("Present_Position")
        return {f"{name}.pos": float(value) for name, value in positions.items()}

    def send_action(self, action: dict) -> dict:
        if not self.bus.is_connected:
            raise RuntimeError("Feetech arm is closed")
        values = {}
        for key, value in action.items():
            if not isinstance(key, str) or not key.endswith(".pos"):
                raise ValueError(f"Unknown action {key}")
            values[key[:-4]] = float(value)
        self.bus.sync_write("Goal_Position", values)
        return dict(action)
