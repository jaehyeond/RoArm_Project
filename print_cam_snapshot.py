"""Bambu P1S 챔버 카메라 스냅샷 — 76th(2026-09-02) 스크래치패드 `bambu_cam.py` 를 repo 로 이전(스크래치패드는 세션마다 소실).

LAN 모드 포트 6000, TLS. 인증 = 80바이트(매직 2 + 'bblp' + access code, 32바이트 널패딩).
프레임마다 16바이트 헤더(앞 4바이트 = payload 크기 LE) + JPEG. DK 원본엔 카메라 구현이 없다.

🔴 76th 실측 함정: 접속 직후 **버퍼된 옛 프레임**이 온다 → 기본 2프레임 받아 **둘째를 저장**한다.
   출력 중엔 프레임 간격이 길어 타임아웃 45 s + 3회 재시도.

사용: python print_cam_snapshot.py <out.jpg> [--ip 192.168.0.96] [--frames 2]
"""
import argparse
import json
import pathlib
import socket
import ssl
import struct
import time

CFG = "/home/cgxr/Documents/DK/DTR/bamboo-3dprinter/config.json"   # access_code 만 읽는다(IP 는 낡음)


def _recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        d = s.recv(min(65536, n - len(buf)))
        if not d:
            raise RuntimeError("수신 중 연결 종료")
        buf += d
    return buf


def grab(ip, access_code, out_path, frames=2, timeout=45):
    auth = struct.pack("<IIII", 0x40, 0x3000, 0, 0) + b"bblp".ljust(32, b"\x00") + access_code.encode().ljust(32, b"\x00")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    img = b""
    with socket.create_connection((ip, 6000), timeout=timeout) as raw, ctx.wrap_socket(raw, server_hostname=ip) as s:
        s.write(auth)
        for _ in range(frames):                       # 앞 프레임은 버린다(옛 프레임 함정)
            size = struct.unpack("<I", _recv_exact(s, 16)[:4])[0]
            if not (1000 < size < 5_000_000):
                raise RuntimeError(f"프레임 크기 이상: {size}")
            img = _recv_exact(s, size)
    if not (img[:2] == b"\xff\xd8" and img[-2:] == b"\xff\xd9"):
        raise RuntimeError("JPEG 마커 불일치")
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_bytes(img)
    return len(img)


def grab_retry(ip, access_code, out_path, frames=2, tries=3, timeout=45):
    last = None
    for _ in range(tries):
        try:
            return grab(ip, access_code, out_path, frames=frames, timeout=timeout)
        except Exception as e:
            last = e
            time.sleep(3)
    raise last


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ip", default="192.168.0.96")
    ap.add_argument("--frames", type=int, default=2)
    a = ap.parse_args()
    code = json.load(open(CFG))["printer"]["access_code"]
    n = grab_retry(a.ip, code, a.out, frames=a.frames)
    print(f"저장 {a.out}  {n} B  (프레임 {a.frames}장 중 마지막)")
