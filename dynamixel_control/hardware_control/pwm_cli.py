import argparse
import serial
import time

# --- USEAGE ---
# Set PWM only:
# python -m hardware_control.pwm_cli 1 90

# Set PWM then read all Dynamixel positions:
# python -m hardware_control.pwm_cli 1 90 --readback

# Just read positions, no PWM command:
# python -m hardware_control.pwm_cli --readback

def read_positions(ser):
    positions = {}
    for sid in [1, 2, 3, 4, 5]:
        ser.reset_input_buffer()
        ser.write(f"G {sid}\n".encode())
        resp = ser.readline().decode().strip()
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

def main():
    parser = argparse.ArgumentParser(description="Control OpenRB-150 PWM servos and read Dynamixel positions")
    parser.add_argument("channel", type=int, choices=[1, 2], nargs="?", help="Servo channel (1=D2, 2=D3)")
    parser.add_argument("angle", type=float, nargs="?", default=180, help="Angle in degrees (0–180), default: 180")
    parser.add_argument("--readback", action="store_true", help="Read back all Dynamixel positions")
    parser.add_argument("--port", default="COM8", help="Serial port (default: COM8)")
    args = parser.parse_args()

    if args.channel is None and not args.readback:
        parser.error("Specify a channel, --readback, or both")

    ser = serial.Serial(args.port, 115200, timeout=1.0)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.readline()  # consume READY

    if args.channel is not None:
        cmd = f"P {args.channel} {int(max(0, min(180, args.angle)))}\n"
        ser.write(cmd.encode())
        resp = ser.readline().decode().strip()
        if resp == "OK":
            print(f"Servo {args.channel} → {args.angle}°")
        else:
            print(f"Error: {resp}")

    if args.readback:
        positions = read_positions(ser)
        print("Dynamixel positions (degrees):")
        print(positions_to_list(positions))

    ser.close()

if __name__ == "__main__":
    main()
