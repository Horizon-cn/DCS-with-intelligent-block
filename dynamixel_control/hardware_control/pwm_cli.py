import argparse
import serial
import time

# --- USAGE ---
# Set PWM only:
# python -m hardware_control.pwm_cli 1 90
#
# Set PWM then read all Dynamixel positions:
# python -m hardware_control.pwm_cli 1 90 --readback
#
# Just read positions, no PWM command:
# python -m hardware_control.pwm_cli --readback


def wait_for_ready(ser: serial.Serial, startup_timeout: float = 10.0) -> bool:
    """Wait for the board to emit READY; return False if it stays silent."""
    deadline = time.monotonic() + startup_timeout
    ser.reset_input_buffer()

    while time.monotonic() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if line == "READY":
            return True

    return False


def send_command(ser: serial.Serial, cmd: str) -> str:
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    return ser.readline().decode(errors="replace").strip()


def read_positions(ser: serial.Serial) -> dict[int, float | None]:
    positions = {}
    for sid in [1, 2, 3, 4, 5]:
        resp = send_command(ser, f"G {sid}")
        if resp.startswith("POS:"):
            try:
                raw = int(resp.split(":")[2])
                positions[sid] = round((raw / 4095.0) * 360.0, 1)
            except (IndexError, ValueError):
                positions[sid] = None
        else:
            positions[sid] = None
    return positions


def positions_to_list(positions: dict[int, float | None]) -> list[float | None]:
    """Return positions ordered by servo ID 1..5 for copy/paste-friendly output."""
    return [positions.get(sid) for sid in [1, 2, 3, 4, 5]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control OpenRB-150 PWM servos and read Dynamixel positions"
    )
    parser.add_argument(
        "channel", type=int, choices=[1, 2], nargs="?", help="Servo channel (1=D2, 2=D3)"
    )
    parser.add_argument(
        "angle", type=float, nargs="?", default=180, help="Angle in degrees (0-180), default: 180"
    )
    parser.add_argument("--readback", action="store_true", help="Read back all Dynamixel positions")
    parser.add_argument("--port", default="COM9", help="Serial port (default: COM9)")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for board startup READY message (default: 10.0)",
    )
    args = parser.parse_args()

    if args.channel is None and not args.readback:
        parser.error("Specify a channel, --readback, or both")

    ser = serial.Serial(args.port, 1000000, timeout=0.25)
    try:
        saw_ready = wait_for_ready(ser, startup_timeout=args.startup_timeout)
        if not saw_ready:
            print(
                f"[WARN] No READY seen from {args.port} after {args.startup_timeout:.1f}s; "
                "continuing because the board may already be running."
            )

        if args.channel is not None:
            resp = send_command(ser, f"P {args.channel} {int(max(0, min(180, args.angle)))}")
            if resp == "OK":
                print(f"Servo {args.channel} -> {args.angle} deg")
            else:
                print(f"Error: {resp}")

        if args.readback:
            positions = read_positions(ser)
            print("Dynamixel positions (degrees):")
            print(positions_to_list(positions))
    finally:
        ser.close()


if __name__ == "__main__":
    main()
