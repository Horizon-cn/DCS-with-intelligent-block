#include <Dynamixel2Arduino.h>
#include <Servo.h>

#define DXL_BUS    Serial1
#define USB        Serial
#define DXL_DIR_PIN -1  // OPENRB-150 handles direction in hardware

Dynamixel2Arduino dxl(DXL_BUS, DXL_DIR_PIN);
using namespace ControlTableItem;

Servo pwmServo1;  // MG90S on D2
Servo pwmServo2;  // MG90S on D3
#define PWM_PIN_1 2
#define PWM_PIN_2 3

String inputBuffer = "";

bool waitForPing(uint8_t id, uint32_t timeoutMs = 1500) {
  uint32_t start = millis();
  while (millis() - start < timeoutMs) {
    if (dxl.ping(id)) return true;
    delay(20);
  }
  return false;
}

bool writeGainVerified(uint16_t item, uint8_t id, int32_t value, uint8_t attempts = 3) {
  for (uint8_t n = 0; n < attempts; n++) {
    if (dxl.writeControlTableItem(item, id, value)) {
      int32_t rb = dxl.readControlTableItem(item, id);
      if (rb == value) return true;
    }
    delay(20);
  }
  return false;
}

bool applyDxlConfig(uint8_t id) {
  if (!waitForPing(id)) return false;

  bool ok = true;
  ok &= dxl.torqueOff(id);
  delay(20);
  ok &= dxl.setOperatingMode(id, OP_POSITION);
  delay(20);

  // IDs 1,2,4,5 use one set of gains; ID 3 uses a different D gain.
  ok &= writeGainVerified(POSITION_P_GAIN, id, 5000);
  ok &= writeGainVerified(POSITION_I_GAIN, id, 0);
  ok &= writeGainVerified(POSITION_D_GAIN, id, (id == 3) ? 10000 : 15000);

  return ok;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);

  USB.begin(115200);
  while (!USB);

  USB.println("[BOOT] USB ready");

  dxl.begin(57600);
  dxl.setPortProtocolVersion(2.0);  // XL430 + XL330 are all Protocol 2.0

  USB.println("[BOOT] Configuring Dynamixels");

  // Initialize all 5 Dynamixel servos
  uint8_t ids[] = {1, 2, 3, 4, 5};
  for (uint8_t id : ids) {
    USB.print("[BOOT] ID ");
    USB.print(id);
    USB.println(" start");
    if (applyDxlConfig(id)) {
      USB.print("[BOOT] ID ");
      USB.print(id);
      USB.println(" OK");
      ledBlink();
    } else {
      USB.print("[BOOT] ID ");
      USB.print(id);
      USB.println(" FAIL");
    }
  }

  // MG90S: 500–2400µs PWM
  USB.println("[BOOT] Configuring PWM servos");
  pwmServo1.attach(PWM_PIN_1, 500, 2400);
  pwmServo2.attach(PWM_PIN_2, 500, 2400);
  pwmServo1.write(110); //default to soft closed
  pwmServo2.write(110); //default to soft closed

  USB.println("READY");
}

void loop() {
  while (USB.available()) {
    char c = (char)USB.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // D <id> <position>  — set Dynamixel goal position (0–4095)
  if (cmd.startsWith("D ")) {
    int id, pos;
    if (sscanf(cmd.c_str(), "D %d %d", &id, &pos) == 2) {
      pos = constrain(pos, 0, 4095);
      USB.println(dxl.setGoalPosition(id, pos) ? "OK" : "ERR:DXL_WRITE");
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  // W <p1> <p2> <p3> <p4> <p5> [pwm1] [pwm2]  — set a full waypoint in one command
  } else if (cmd.startsWith("W ")) {
    int p1, p2, p3, p4, p5, pwm1, pwm2;
    int count = sscanf(cmd.c_str(), "W %d %d %d %d %d %d %d", &p1, &p2, &p3, &p4, &p5, &pwm1, &pwm2);
    if (count == 5 || count == 6 || count == 7) {
      bool ok = true;
      int positions[] = {p1, p2, p3, p4, p5};
      uint8_t ids[] = {1, 2, 3, 4, 5};
      for (int i = 0; i < 5; i++) {
        positions[i] = constrain(positions[i], 0, 4095);
        ok &= dxl.setGoalPosition(ids[i], positions[i]);
      }

      if (count >= 6) {
        pwm1 = constrain(pwm1, 0, 180);
        pwmServo1.write(pwm1);
      }
      if (count >= 7) {
        pwm2 = constrain(pwm2, 0, 180);
        pwmServo2.write(pwm2);
      }

      USB.println(ok ? "OK" : "ERR:DXL_WRITE");
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  // G <id>  — get present position
  } else if (cmd.startsWith("G ")) {
    int id;
    if (sscanf(cmd.c_str(), "G %d", &id) == 1) {
      int32_t pos = dxl.getPresentPosition(id);
      USB.print("POS:"); USB.print(id);
      USB.print(":"); USB.println(pos);
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  // P <channel> <angle>  — PWM servo, channel 1=D2, 2=D3, angle 0–180
  } else if (cmd.startsWith("P ")) {
    int ch, angle;
    if (sscanf(cmd.c_str(), "P %d %d", &ch, &angle) == 2) {
      angle = constrain(angle, 0, 180);
      if      (ch == 1) pwmServo1.write(angle);
      else if (ch == 2) pwmServo2.write(angle);
      else { USB.println("ERR:BAD_CHANNEL"); return; }
      USB.println("OK");
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  // K <id>  — ping a Dynamixel
  } else if (cmd.startsWith("K ")) {
    int id;
    if (sscanf(cmd.c_str(), "K %d", &id) == 1) {
      USB.print("PING:"); USB.print(id);
      USB.println(dxl.ping(id) ? ":OK" : ":FAIL");
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  // T <id> <0|1>  — torque off/on
  } else if (cmd.startsWith("T ")) {
    int id, state;
    if (sscanf(cmd.c_str(), "T %d %d", &id, &state) == 2) {
      state ? dxl.torqueOn(id) : dxl.torqueOff(id);
      USB.println("OK");
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  // R <id>  — reboot a Dynamixel (clears hardware error shutdown states)
  } else if (cmd.startsWith("R ")) {
    int id;
    if (sscanf(cmd.c_str(), "R %d", &id) == 1) {
      if (!dxl.reboot(id)) {
        USB.println("ERR:DXL_REBOOT");
      } else {
        delay(400);  // allow reboot to complete before reconfiguring registers
        USB.println(applyDxlConfig(id) ? "OK" : "ERR:DXL_CFG");
      }
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  // A <id>  — apply operating mode + PID configuration without reboot
  } else if (cmd.startsWith("A ")) {
    int id;
    if (sscanf(cmd.c_str(), "A %d", &id) == 1) {
      USB.println(applyDxlConfig(id) ? "OK" : "ERR:DXL_CFG");
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  // C <id>  — read PID gains, returns PID:<id>:<P>:<I>:<D>
  } else if (cmd.startsWith("C ")) {
    int id;
    if (sscanf(cmd.c_str(), "C %d", &id) == 1) {
      if (!dxl.ping(id)) {
        USB.println("ERR:DXL_PING");
      } else {
        int32_t p = dxl.readControlTableItem(POSITION_P_GAIN, id);
        int32_t i = dxl.readControlTableItem(POSITION_I_GAIN, id);
        int32_t d = dxl.readControlTableItem(POSITION_D_GAIN, id);
        USB.print("PID:"); USB.print(id);
        USB.print(":"); USB.print(p);
        USB.print(":"); USB.print(i);
        USB.print(":"); USB.println(d);
      }
    } else {
      USB.println("ERR:BAD_ARGS");
    }

  } else {
    USB.println("ERR:UNKNOWN_CMD");
  }

  ledBlink();
}

void ledBlink() {
  static bool ledState = false;
  ledState = !ledState;
  digitalWrite(LED_BUILTIN, ledState ? HIGH : LOW);
}
