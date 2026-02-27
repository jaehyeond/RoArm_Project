#!/usr/bin/env python3
"""
Chrome Remote Desktop Black Screen Fix
=======================================
Windows → Linux 원격 접속 시 검은 화면 문제 해결.

원인: CRD가 새 가상 X 서버(:20+)를 만들어서 기존 세션(:1)에 접속 못함.
해결: 기존 X 세션(:1)을 재사용하도록 수정.

사용법:
  sudo python3 fix_chrome_remote_desktop.py

수정 내용:
  1. FIRST_X_DISPLAY_NUMBER = 20 → 1
  2. get_unused_display_number() → 항상 :1 반환
  3. XDesktop.launch_session() → 기존 X 세션 재사용
  4. XDesktop.launch_desktop_session() → no-op (기존 세션 사용)
"""

import os
import sys
import shutil

CRD_PATH = "/opt/google/chrome-remote-desktop/chrome-remote-desktop"
BACKUP_PATH = CRD_PATH + ".orig"


def main():
    if os.geteuid() != 0:
        print("ERROR: root 권한 필요. sudo python3 fix_chrome_remote_desktop.py")
        sys.exit(1)

    # Step 1: Backup
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(CRD_PATH, BACKUP_PATH)
        print(f"[OK] 백업 완료: {BACKUP_PATH}")
    else:
        print(f"[SKIP] 백업 이미 존재: {BACKUP_PATH}")

    # Step 2: Read file
    with open(CRD_PATH, 'r') as f:
        content = f.read()

    original_content = content

    # =========================================================
    # Fix 1: FIRST_X_DISPLAY_NUMBER = 20 → 1
    # =========================================================
    old_1 = "FIRST_X_DISPLAY_NUMBER = 20"
    new_1 = "FIRST_X_DISPLAY_NUMBER = 1"
    if old_1 in content:
        content = content.replace(old_1, new_1)
        print("[OK] Fix 1: FIRST_X_DISPLAY_NUMBER = 1")
    elif new_1 in content:
        print("[SKIP] Fix 1: 이미 적용됨")
    else:
        print("[WARN] Fix 1: FIRST_X_DISPLAY_NUMBER 찾을 수 없음")

    # =========================================================
    # Fix 2: get_unused_display_number() → 항상 FIRST_X_DISPLAY_NUMBER 반환
    # =========================================================
    old_2 = '''  @staticmethod
  def get_unused_display_number():
    """Return a candidate display number for which there is currently no
    X Server lock file"""
    display = FIRST_X_DISPLAY_NUMBER
    while os.path.exists(X_LOCK_FILE_TEMPLATE % display):
      display += 1
    return display'''

    new_2 = '''  @staticmethod
  def get_unused_display_number():
    """Return a candidate display number for which there is currently no
    X Server lock file"""
    # [CRD Fix] 기존 X 세션 재사용 - 항상 현재 디스플레이 번호 반환
    # display = FIRST_X_DISPLAY_NUMBER
    # while os.path.exists(X_LOCK_FILE_TEMPLATE % display):
    #   display += 1
    return FIRST_X_DISPLAY_NUMBER'''

    if old_2 in content:
        content = content.replace(old_2, new_2)
        print("[OK] Fix 2: get_unused_display_number() → 항상 :1 반환")
    elif "# [CRD Fix] 기존 X 세션 재사용" in content:
        print("[SKIP] Fix 2: 이미 적용됨")
    else:
        print("[WARN] Fix 2: get_unused_display_number() 패턴 매칭 실패")

    # =========================================================
    # Fix 3: XDesktop.launch_session() → 기존 X 세션 재사용
    # =========================================================
    old_3 = '''  def launch_session(self, *args, **kwargs):
    logging.info("Launching X server and X session.")
    super(XDesktop, self).launch_session(*args, **kwargs)'''

    new_3 = '''  def launch_session(self, server_args, backoff_time):
    # [CRD Fix] 기존 X 세션(:1) 재사용 — 새 X 서버/세션 생성 안 함
    logging.info("Reusing existing X session instead of launching new one.")

    # gnubby 설정은 유지
    self._setup_gnubby()

    # 새 X 서버 시작 대신 기존 디스플레이에 연결
    display = self.get_unused_display_number()
    self.child_env["DISPLAY"] = ":%d" % display
    self.child_env["XAUTHORITY"] = os.path.expanduser("~/.Xauthority")

    logging.info("Connected to existing display :%d" % display)

    # 기존 데스크탑 세션 사용 — launch_desktop_session() 호출 안 함
    # pre-session 스크립트도 실행
    self._launch_pre_session()

    # inhibitor 기록 (프로세스 재시작 관리용)
    self.server_inhibitor.record_started(MINIMUM_PROCESS_LIFETIME,
                                        backoff_time)
    self.session_inhibitor.record_started(MINIMUM_PROCESS_LIFETIME,
                                       backoff_time)'''

    if old_3 in content:
        content = content.replace(old_3, new_3)
        print("[OK] Fix 3: XDesktop.launch_session() → 기존 세션 재사용")
    elif "# [CRD Fix] 기존 X 세션(:1) 재사용" in content:
        print("[SKIP] Fix 3: 이미 적용됨")
    else:
        print("[WARN] Fix 3: XDesktop.launch_session() 패턴 매칭 실패")

    # =========================================================
    # Fix 4: XDesktop.launch_desktop_session() → no-op
    # =========================================================
    old_4 = '''  def launch_desktop_session(self):
    # Start desktop session.
    # The /dev/null input redirection is necessary to prevent the X session
    # reading from stdin.  If this code runs as a shell background job in a
    # terminal, any reading from stdin causes the job to be suspended.
    # Daemonization would solve this problem by separating the process from the
    # controlling terminal.
    xsession_command = choose_x_session()
    if xsession_command is None:
      raise Exception("Unable to choose suitable X session command.")

    logging.info("Launching X session: %s" % xsession_command)
    self.session_proc = subprocess.Popen(xsession_command,
                                         stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         cwd=HOME_DIR,
                                         env=self.child_env)

    if not self.session_proc.pid:
      raise Exception("Could not start X session")

    output_filter_thread = SessionOutputFilterThread(self.session_proc.stdout,
        "Session output: ", SESSION_OUTPUT_TIME_LIMIT_SECONDS)
    output_filter_thread.start()'''

    new_4 = '''  def launch_desktop_session(self):
    # [CRD Fix] 기존 데스크탑 세션 재사용 — 새 세션 시작 안 함
    logging.info("Reusing existing desktop session (no-op).")
    # xsession_command = choose_x_session()
    # if xsession_command is None:
    #   raise Exception("Unable to choose suitable X session command.")
    #
    # logging.info("Launching X session: %s" % xsession_command)
    # self.session_proc = subprocess.Popen(xsession_command,
    #                                      stdin=subprocess.DEVNULL,
    #                                      stdout=subprocess.PIPE,
    #                                      stderr=subprocess.STDOUT,
    #                                      cwd=HOME_DIR,
    #                                      env=self.child_env)
    #
    # if not self.session_proc.pid:
    #   raise Exception("Could not start X session")
    #
    # output_filter_thread = SessionOutputFilterThread(self.session_proc.stdout,
    #     "Session output: ", SESSION_OUTPUT_TIME_LIMIT_SECONDS)
    # output_filter_thread.start()
    pass'''

    # Need to find the XDesktop version specifically (after line ~1770)
    # The base Desktop class also has launch_desktop_session at line ~1156
    # We only want to modify the XDesktop one
    if old_4 in content:
        # Find the last occurrence (XDesktop's version comes after Desktop's)
        # Actually both might match. Let's check if old_4 appears in XDesktop context
        idx = content.rfind(old_4)
        if idx != -1:
            content = content[:idx] + new_4 + content[idx + len(old_4):]
            print("[OK] Fix 4: XDesktop.launch_desktop_session() → no-op")
        else:
            print("[WARN] Fix 4: rfind 실패")
    elif "# [CRD Fix] 기존 데스크탑 세션 재사용" in content:
        print("[SKIP] Fix 4: 이미 적용됨")
    else:
        print("[WARN] Fix 4: XDesktop.launch_desktop_session() 패턴 매칭 실패")

    # =========================================================
    # Write
    # =========================================================
    if content == original_content:
        print("\n변경 사항 없음.")
        return

    with open(CRD_PATH, 'w') as f:
        f.write(content)
    print(f"\n[DONE] 수정 완료: {CRD_PATH}")
    print("\n다음 단계:")
    print("  sudo systemctl restart chrome-remote-desktop@$(whoami).service")
    print("\n복원 방법:")
    print(f"  sudo cp {BACKUP_PATH} {CRD_PATH}")
    print("  sudo systemctl restart chrome-remote-desktop@$(whoami).service")


if __name__ == "__main__":
    main()
