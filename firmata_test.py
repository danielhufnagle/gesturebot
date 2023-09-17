import pyfirmata
import time

port = '/dev/cu.usbmodem1301'
board = pyfirmata.Arduino(port)

while True:
    try:
        board.digital[13].write(1)
        time.sleep(0.5)
        board.digital[13].write(0)
        time.sleep(0.5)
    except KeyboardInterrupt:
        break
