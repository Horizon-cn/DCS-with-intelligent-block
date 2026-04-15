import serial
import time
from enum import IntEnum
from typing import Callable, Sequence

class Gripper(IntEnum):
    OPEN = 180
    LOCK = 90


PidGains = tuple[int, int, int]

class OpenRB150:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.ser.reset_input_buffer()
        self._torque_enabled = set()  # track which IDs have torque on

    def _send(self, cmd: str) -> str:
        self.ser.write((cmd + '\n').encode())
        return self.ser.readline().decode(errors="replace").strip()

    def _send_with_retry(
        self,
        cmd: str,
        is_success: Callable[[str], bool],
        attempts: int = 3,
        timeout: float | None = None,
    ) -> tuple[bool, str]:
        """Send command up to `attempts` times until response passes `is_success`."""
        last_resp = ""
        previous_timeout = self.ser.timeout
        try:
            if timeout is not None:
                self.ser.timeout = timeout
            for _ in range(attempts):
                resp = self._send(cmd)
                last_resp = resp
                if is_success(resp):
                    return True, resp
            return False, last_resp
        finally:
            self.ser.timeout = previous_timeout

    # --- Dynamixel ---

    def set_position(self, dxl_id: int, position: int) -> bool:
        """Raw position units (0–4095). Enables torque on first call per ID."""
        if dxl_id not in self._torque_enabled:
            if not self.torque(dxl_id, True):
                return False
            self._torque_enabled.add(dxl_id)
        ok, _ = self._send_with_retry(
            f"D {dxl_id} {max(0, min(4095, position))}",
            lambda resp: resp == "OK",
        )
        return ok

    def set_angle_deg(self, dxl_id: int, degrees: float) -> bool:
        """
        Degrees to raw units. XL430 and XL330 both have 360° range.
        """
        raw = int((degrees / 360.0) * 4095)
        return self.set_position(dxl_id, raw)

    def get_position(self, dxl_id: int) -> int | None:
        ok, resp = self._send_with_retry(
            f"G {dxl_id}",
            lambda r: r.startswith("POS:"),
        )
        if not ok:
            return None
        if resp.startswith("POS:"):
            try:
                return int(resp.split(":")[2])
            except (IndexError, ValueError):
                pass
        return None

    def get_angle_deg(self, dxl_id: int) -> float | None:
        raw = self.get_position(dxl_id)
        return round((raw / 4095.0) * 360.0, 1) if raw is not None else None

    def ping(self, dxl_id: int) -> bool:
        ok, _ = self._send_with_retry(
            f"K {dxl_id}",
            lambda resp: resp == f"PING:{dxl_id}:OK",
        )
        return ok

    def torque(self, dxl_id: int, enable: bool) -> bool:
        ok, _ = self._send_with_retry(
            f"T {dxl_id} {1 if enable else 0}",
            lambda resp: resp == "OK",
        )
        if ok and not enable:
            self._torque_enabled.discard(dxl_id)
        return ok

    def reboot(self, dxl_id: int) -> bool:
        """Reboot a Dynamixel to clear shutdown/hardware error states."""
        ok, _ = self._send_with_retry(
            f"R {dxl_id}",
            lambda resp: resp == "OK",
        )
        if ok:
            self._torque_enabled.discard(dxl_id)
            time.sleep(0.2)
        return ok

    def reset_all_dynamixels(self, ids: list[int] | None = None) -> bool:
        """Reboot all configured Dynamixels and return True only if all succeed."""
        if ids is None:
            ids = [1, 2, 3, 4, 5]

        all_ok = True
        for sid in ids:
            ok = self.reboot(sid)
            if ok:
                print(f"  Reset ID {sid}: OK")
            else:
                print(f"  Reset ID {sid}: FAIL")
                all_ok = False
        return all_ok

    def apply_config(self, dxl_id: int) -> bool:
        """Apply firmware-side operating mode + PID settings without reboot."""
        ok, _ = self._send_with_retry(
            f"A {dxl_id}",
            lambda resp: resp == "OK",
        )
        if ok:
            self._torque_enabled.discard(dxl_id)
        return ok

    def apply_config_all_dynamixels(self, ids: list[int] | None = None) -> bool:
        """Apply firmware-side operating mode + PID settings to all configured IDs."""
        if ids is None:
            ids = [1, 2, 3, 4, 5]

        all_ok = True
        for sid in ids:
            ok = self.apply_config(sid)
            if ok:
                print(f"  Config ID {sid}: OK")
            else:
                print(f"  Config ID {sid}: FAIL")
                all_ok = False
        return all_ok

    def get_pid_gains(self, dxl_id: int) -> PidGains | None:
        """Read (P, I, D) gains from one Dynamixel."""
        ok, resp = self._send_with_retry(
            f"C {dxl_id}",
            lambda r: r.startswith(f"PID:{dxl_id}:"),
        )
        if not ok:
            return None

        try:
            _, _, p_str, i_str, d_str = resp.split(":")
            return (int(p_str), int(i_str), int(d_str))
        except (ValueError, IndexError):
            return None

    def get_all_pid_gains(self, ids: list[int] | None = None) -> dict[int, PidGains | None]:
        """Read PID gains for all configured Dynamixels."""
        if ids is None:
            ids = [1, 2, 3, 4, 5]
        return {sid: self.get_pid_gains(sid) for sid in ids}
    
    def read_all_positions(self) -> dict[int, float | None]:
        """
        Read current angles (in degrees) for all 5 Dynamixel IDs.
        Returns {id: angle_deg} — None for any servo that doesn't respond.
        """
        return {sid: self.get_angle_deg(sid) for sid in [1, 2, 3, 4, 5]}

    def read_all_positions_list(self) -> list[float | None]:
        """Read all positions as [ID1, ID2, ID3, ID4, ID5] for easy copy/paste."""
        positions = self.read_all_positions()
        return [positions[sid] for sid in [1, 2, 3, 4, 5]]

    # --- PWM (MG90S) ---

    def set_pwm_angle(self, channel: int, angle: float = 180) -> bool:
        """channel: 1 = D2, 2 = D3. angle: 0–180°"""
        ok, _ = self._send_with_retry(
            f"P {channel} {int(max(0, min(180, angle)))}",
            lambda resp: resp == "OK",
        )
        return ok

    def close(self):
        self.ser.close()


    ### --- Helper Functions ---

    @staticmethod
    def _degrees_to_raw(degrees: float) -> int:
        return max(0, min(4095, int((degrees / 360.0) * 4095)))

    def send_waypoint_batch(
        self,
        waypoint: Sequence[float | int | None],
        *,
        attempts: int = 1,
        timeout: float = 0.02,
    ) -> bool:
        """
        Send a full waypoint in one board command.
        Motion streaming uses a single best-effort attempt by default.
        """
        assert len(waypoint) >= 5, "Waypoint must have at least 5 Dynamixel values"

        dxl_raw = [self._degrees_to_raw(float(angle)) for angle in waypoint[:5]]
        parts = ["W", *[str(raw) for raw in dxl_raw]]

        ok, _ = self._send_with_retry(
            " ".join(parts),
            lambda resp: resp == "OK",
            attempts=attempts,
            timeout=timeout,
        )
        return ok

    def execute_waypoint(self, waypoint: list, delay: float = 0.5) -> bool:
        """
        Send a waypoint: [ID1, ID2, ID3, ID4, ID5, PWM1 (optional), PWM2 (optional)]
        Omit PWM values (or pass None) to leave gripper state unchanged.
        """
        assert len(waypoint) >= 5, "Waypoint must have at least 5 Dynamixel values"
        ok = self.send_waypoint_batch(waypoint)
        pwm_angs = list(waypoint[5:7])
        while len(pwm_angs) < 2:
            pwm_angs.append(None)
        for channel, angle in enumerate(pwm_angs, start=1):
            if angle is not None:
                ok &= self.set_pwm_angle(channel, float(angle))
        time.sleep(delay)
        return ok

    def execute_trajectory(
        self,
        trajectory: list[list[float]],
        delay: float = 0.5,
        readback: bool = False,
        log_steps: bool = False,
    ):
        """
        Execute a list of waypoints in sequence.
        delay: seconds between each waypoint.
        readback: if True, print current servo angles after each waypoint
        """
        next_deadline = time.perf_counter()
        dropped_steps = 0
        previous_pwm = [None, None]
        for i, waypoint in enumerate(trajectory):
            if log_steps:
                print(f"  Step {i+1}/{len(trajectory)}: {waypoint}")
            ok = self.send_waypoint_batch(waypoint)
            if not ok:
                dropped_steps += 1

            current_pwm = list(waypoint[5:7])
            while len(current_pwm) < 2:
                current_pwm.append(None)
            for channel, angle in enumerate(current_pwm, start=1):
                if angle is not None and angle != previous_pwm[channel - 1]:
                    self.set_pwm_angle(channel, float(angle))
            previous_pwm = current_pwm

            if readback:
                positions = self.read_all_positions_list()
                print(f"  Readback: {positions}")
            if delay > 0:
                next_deadline += delay
                remaining = next_deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
        if dropped_steps:
            print(f"[WARN] Dropped or timed out on {dropped_steps}/{len(trajectory)} steps")

if __name__ == "__main__":
    bot = OpenRB150(port="COM9")
    ids = [1, 2, 3, 4, 5]

    print("Resetting Dynamixels...")
    if not bot.reset_all_dynamixels(ids):
        print("[WARN] One or more reset commands failed")

    print("Applying Dynamixel config...")
    if not bot.apply_config_all_dynamixels(ids):
        print("[WARN] One or more config commands failed")

    print("Reading PID gains...")
    gains = bot.get_all_pid_gains(ids)
    for sid in ids:
        gid = gains[sid]
        if gid is None:
            print(f"  ID {sid}: PID READ FAIL")
        else:
            p_gain, i_gain, d_gain = gid
            print(f"  ID {sid}: P={p_gain} I={i_gain} D={d_gain}")

    #Set grippers to soft closed
    bot.set_pwm_angle(1, 90)
    bot.set_pwm_angle(2, 90)

    #Ping dynamixels, error out if any are missing (IDs 1–5)
    print("Pinging servos...")
    for sid in ids:
        if bot.ping(sid):
            print(f"  ID {sid}: OK")
        else:
            print(f"  ID {sid}: NOT FOUND — aborting")
            bot.close()
            raise SystemExit(1)
        
    #read back positions
    positions = bot.read_all_positions_list()
    print(positions)

    # STRAIGHT PLANAR MOVEMENTS
    # [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
    # [180, 259.5, 71.1, 161.7, 180],  #1 PLANAR 1st LIFTED (BATTERY INTERFERENCE?) CHECK <---------
    # [180, 182.0, 77.7, 254.7, 180],  #1 PLANAR 2nd LIFTED

    # [180.6, 247.4, 53.3, 165.6, 183.7],  #1<>2 PLANAR TRANSITION, 1st LIFTED, PART 1
    # [181.6, 215.3, 16.1, 162.0, 185.1],  #1<>2 PLANAR TRANSITION, 1st LIFTED, PART 2

    # [180, 167.7, 60.7, 254.8, 180],  #1<>2 PLANAR TRANSITION, 2nd LIFTED, PART 1
    # [180, 164.3, 20.5, 216.8, 180],  #1<>2 PLANAR TRANSITION, 2nd LIFTED, PART 2

    # [180, 182.6, 3.5, 180.6, 180, Gripper.LOCK, Gripper.LOCK],  #2 PLANAR BOTH LOCKED
    # [180, 193.6, -3.9, 164.2, 180],  #2 PLANAR 1st LIFTED
    # [180, 166.2, -3.8, 191.2, 180],  #2 PLANAR 2nd LIFTED

    move_1_forward_planar = [ #TESTED WORKING 4/6/26
        [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
        [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.OPEN],  #1 PLANAR 2nd OPEN
        [180, 182.0, 77.7, 254.7, 180],  #1 PLANAR 2nd LIFTED
        [180, 167.7, 60.7, 254.8, 180],  #1<>2 PLANAR TRANSITION, 2nd LIFTED, PART 1
        [180, 164.3, 20.5, 216.8, 180],  #1<>2 PLANAR TRANSITION, 2nd LIFTED, PART 2
        [180, 166.2, -3.8, 191.2, 180],  #2 PLANAR 2nd LIFTED
        [180, 182.6, 3.5, 180.6, 180, Gripper.LOCK, Gripper.OPEN],  #2 PLANAR 2nd OPEN
        [180, 182.6, 3.5, 180.6, 180, Gripper.LOCK, Gripper.LOCK],  #2 PLANAR BOTH LOCKED
        [180, 182.6, 3.5, 180.6, 180, Gripper.OPEN, Gripper.LOCK],  #2 PLANAR 1st OPEN
        [180, 193.6, -3.9, 164.2, 180],  #2 PLANAR 1st LIFTED
        [181.6, 215.3, 16.1, 162.0, 185.1],  #1<>2 PLANAR TRANSITION, 1st LIFTED, PART 2
        [180.6, 247.4, 53.3, 165.6, 183.7],  #1<>2 PLANAR TRANSITION, 1st LIFTED, PART 1
        [180, 259.5, 71.1, 161.7, 180],  #1 PLANAR 1st LIFTED
        [180, 224.6, 86.95, 222.6, 180, Gripper.OPEN, Gripper.LOCK],  #1 PLANAR FLAT
        [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
    ]

    move_1_backward_planar = [ #TESTED WORKING 4/6/26
        [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
        [180, 224.6, 86.95, 222.6, 180, Gripper.OPEN, Gripper.LOCK],  #1 PLANAR 1st OPEN
        [180, 259.5, 71.1, 161.7, 180],  #1 PLANAR 1st LIFTED
        [180.6, 247.4, 53.3, 165.6, 183.7],  #1<>2 PLANAR TRANSITION, 1st LIFTED, PART 1
        [181.6, 215.3, 16.1, 162.0, 185.1],  #1<>2 PLANAR TRANSITION, 1st LIFTED, PART 2
        [180, 193.6, -3.9, 164.2, 180],  #2 PLANAR 1st LIFTED
        [180, 182.6, 3.5, 180.6, 180, Gripper.OPEN, Gripper.LOCK],  #2 PLANAR 1st OPEN
        [180, 182.6, 3.5, 180.6, 180, Gripper.LOCK, Gripper.LOCK],  #2 PLANAR BOTH LOCKED
        [180, 182.6, 3.5, 180.6, 180, Gripper.LOCK, Gripper.OPEN],  #2 PLANAR 2nd OPEN
        [180, 166.2, -3.8, 191.2, 180],  #2 PLANAR 2nd LIFTED
        [180, 164.3, 20.5, 216.8, 180],  #1<>2 PLANAR TRANSITION, 2nd LIFTED, PART 2
        [180, 167.7, 60.7, 254.8, 180],  #1<>2 PLANAR TRANSITION, 2nd LIFTED, PART 1
        [180, 182.0, 77.7, 254.7, 180],  #1 PLANAR 2nd LIFTED
        [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.OPEN],  #1 PLANAR 2nd OPEN
        [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
    ]

    move_1_up_forward_from_planar = [
        [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
        [180, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.OPEN],  #1 PLANAR 2nd OPEN
        [180, 182.0, 77.7, 254.7, 180],  #1 PLANAR 2nd LIFTED
        [90, 182.0, 77.7, 254.7, 180],  #1 PLANAR 2nd LIFTED 2nd STEER -90
        [90, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.OPEN],  #1 PLANAR 2nd OPEN 2nd STEER -90
        [90, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED 2nd STEER -90
        [90, 224.6, 86.95, 222.6, 180, Gripper.OPEN, Gripper.LOCK],  #1 PLANAR 1st OPEN 2nd STEER -90
        [90, 259.5, 71.1, 161.7, 180],  #1 PLANAR 1st LIFTED
        [180, 259.5, 71.1, 161.7, 90],  #1 PLANAR 1st LIFTED 2nd STEER 0, 1st STEER +90
        [180, 269.4, 21.5, 129.6, 90], #1 UP FORWARD, 1st LIFTED, 2nd STEER 0, 1st STEER +90
        [270, 269.4, 21.5, 129.6, 90], #1 UP FORWARD, 1st LIFTED, 2nd STEER +90, 1st STEER +90
        [270, 264.6, 38.6, 130.9, 90, Gripper.OPEN, Gripper.LOCK],  #1 UP 1st OPEN, 2nd STEER +90, 1st STEER +90
        [270, 264.7, 32.1, 125.9, 90, Gripper.LOCK, Gripper.LOCK],  #1 UP BOTH LOCKED, 2nd STEER +90, 1st STEER +90
    ]

    move4 = [
        [270, 264.7, 32.1, 125.9, 90, Gripper.LOCK, Gripper.LOCK],  #1 UP BOTH LOCKED, 2nd STEER +90, 1st STEER +90
        [270, 264.7, 32.1, 125.9, 90, Gripper.LOCK, Gripper.OPEN],  #1 UP BOTH LOCKED, 2nd STEER +90, 1st STEER +90
        [270, 259.5, 71.1, 161.7, 90],  #1 PLANAR 1st LIFTED (BATTERY INTERFERENCE?)
        [180, 259.5, 71.1, 161.7, 180],  #1 PLANAR 1st LIFTED (BATTERY INTERFERENCE?)
        [180, 136.8, 46.2, 175.7, 180], #down right lifted
        [180, 166.9, 66.5, 164.7, 180], #down right planted (+2 preload J3)
        [180, 166.9, 63.5, 164.7, 180, Gripper.LOCK, Gripper.LOCK], #down right planted, both locked
    ]

    move5 = [
        [180, 166.9, 63.5, 164.7, 180, Gripper.LOCK, Gripper.LOCK], #down right planted, both locked
        [180, 166.9, 63.5, 164.7, 180, Gripper.OPEN, Gripper.LOCK], #down right planted, both locked 1st open
        [180, 168.4, 40.4, 139.0, 180], #down right 1st lifted
        [180, 259.5, 71.1, 161.7, 180],  #1 PLANAR 1st LIFTED
        [90, 259.5, 71.1, 161.7, 180],  #1 PLANAR 1st LIFTED, 1st STEER -90
        [90, 222.6, 86.95, 222.6, 180],  #1 PLANAR PLANTED (+2 preload J2)
        [90, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
    ]

    move6 = [
        [90, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
        [90, 224.6, 86.95, 222.6, 180, Gripper.LOCK, Gripper.OPEN],  #1 PLANAR 2nd OPEN
        [90, 183.0, 81.3, 257.6, 180],  #1 PLANAR 2nd LIFTED
        [90, 183.0, 81.3, 257.6, 90],  #1 PLANAR 2nd LIFTED, STEER 2 -90
        [90, 224.0, 89.6, 218.1, 90],  #1 PLANAR 2nd PLANTED (+2 preload J5)
        [90, 224.6, 86.95, 220.6, 90, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
    ]

    move7 = [
        [90, 224.6, 86.95, 222.6, 90, Gripper.LOCK, Gripper.LOCK],  #1 PLANAR BOTH LOCKED
        [90, 224.6, 86.95, 222.6, 90, Gripper.OPEN, Gripper.LOCK],  #1 PLANAR 1st OPEN
        [90, 265.8, 77.3, 163.4, 90],  #1 PLANAR 1st LIFTED, 1st steer -90
        [270, 269.6, 79.1, 173.9, 180],  #1 PLANAR 1st LIFTED, 1st steer +90
        [270, 168.9, 39.9, 141.5, 180],  #1 down 1st lifted
        [270, 165.4, 62.4, 164.8, 180],  #1 down planted
        [270, 165.4, 62.4, 164.8, 180, Gripper.LOCK, Gripper.LOCK],  #1 down planted both locked
    ]

    delay = 0.5  # seconds between waypoints

    print("Executing trajectory...")
    for sid in ids:
        bot.torque(sid, True)
    
    print("Moving 1 forward...")
    bot.execute_trajectory(move_1_forward_planar, delay=0.25, readback=True)    

    for sid in ids:
        bot.torque(sid, False)
    #Set grippers to soft closed
    bot.set_pwm_angle(1, 90)
    bot.set_pwm_angle(2, 100)

    print("Finished, exiting.")
    bot.close()
