import sys, os, time, threading
import termios, tty
sys.path.append(os.path.abspath("."))
import cli

def send_keys():
    time.sleep(1)
    # Write a TAB directly to the terminal's stdin fd (if we could), but we can just mock get_key
    pass

# We will patch get_key to yield some keys
old_get_key = cli.get_key

key_sequence = ["TAB", "TAB", "TAB", "SPACE"]
key_idx = 0

def mock_get_key(fd):
    global key_idx
    time.sleep(0.5)
    if key_idx < len(key_sequence):
        k = key_sequence[key_idx]
        key_idx += 1
        return k
    return "CTRL_C"

cli.get_key = mock_get_key

try:
    cli.main()
except Exception as e:
    import traceback
    with open("crash.log", "w") as f:
        traceback.print_exc(file=f)
