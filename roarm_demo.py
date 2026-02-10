from roarm_sdk import RoArm
import time, json

SAFE_ACC = 300
SAFE_SPEED = 900
HOME = [0,0,0,0,0,0]

class Arm:
    def __init__(self, port="auto"):
        self.arm = RoArm(port=port)

    def torque(self, on:int):
        # 중요: 1/0 (bool 아님)
        self.arm.torque_set(on)

    def home(self, acc=SAFE_ACC, speed=SAFE_SPEED):
        self.arm.joints_angle_ctrl(angles=HOME, acc=acc, speed=speed)

    def move(self, angles, acc=SAFE_ACC, speed=SAFE_SPEED):
        if len(angles) != 6:
            raise ValueError("angles must be length 6 (J1..J6)")
        self.arm.joints_angle_ctrl(angles=angles, acc=acc, speed=speed)

    # (확실하지 않음) SDK에 현재 각도 읽기 함수가 있을 경우 사용
    def get_angles_safe(self):
        if hasattr(self.arm, "get_angles"):
            return list(self.arm.get_angles())
        # 미구현이면 None 반환
        return None

    # (확실하지 않음) RAW JSON 커맨드 전송 지원 시 사용: 예) arm.send(json_str) 같은 메서드
    def send_raw_json(self, payload:dict):
        if hasattr(self.arm, "send"):
            return self.arm.send(json.dumps(payload))
        elif hasattr(self.arm, "raw_command"):
            return self.arm.raw_command(json.dumps(payload))
        else:
            raise NotImplementedError("raw JSON command is not supported in this SDK build.")
