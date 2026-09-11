"""Linux-only candidate transport fix after observed open-time ESP32 reset.

Deassert RTS and DTR in one ioctl instead of pySerial's two sequential calls.
This removes that software-created intermediate state; it cannot guarantee
that the kernel, USB driver, or board produces no opening/closing transient.
No device opens at import. No installed SDK/package is changed.
"""
import fcntl
import struct
import termios
import serial

class AtomicInactiveSerial(serial.Serial):
    def _update_dtr_state(self):
        if self._dtr_state or self._rts_state:
            raise RuntimeError('This transport requires both DTR and RTS inactive before open')
        fcntl.ioctl(self.fd, termios.TIOCMBIC,
                    struct.pack('I', termios.TIOCM_DTR | termios.TIOCM_RTS))

    def _update_rts_state(self):
        self._update_dtr_state()

def is_boot_text(text):
    return any(marker in text for marker in ('ets Jul', 'rst:0x', 'POWERON_RESET', 'SPI_FAST_FLASH_BOOT'))
