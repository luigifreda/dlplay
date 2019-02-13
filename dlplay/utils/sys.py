# ******************************************************************************
# This file is part of dlplay
# 
# Copyright (C) Luigi Freda <luigi dot freda at gmail dot com>
# 
# dlplay is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dlplay is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dlplay. If not, see <http://www.gnu.org/licenses/>.
# ******************************************************************************
import threading
import multiprocessing as mp
import traceback
import os
import signal
import time


def force_kill_all_and_exit(code=0, verbose=True):
    # Log active threads (excluding main)
    active_threads = [t for t in threading.enumerate() if t != threading.main_thread()]
    if active_threads:
        if verbose:
            print(f"[!] Active threads: {[t.name for t in active_threads]}")

    # Attempt to stop threads (cannot forcibly kill threads in Python)
    for t in active_threads:
        if verbose:
            print(f"[!] Thread {t.name} is still running and cannot be force-killed.")

    # Terminate all active multiprocessing children
    active_children = mp.active_children()
    if active_children:
        if verbose:
            print(f"[!] Active child processes: {active_children}")
    for p in active_children:
        try:
            if verbose:
                print(f"[!] Terminating process PID {p.pid}...")
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                if verbose:
                    print(f"[!] Killing stubborn process PID {p.pid}...")
                os.kill(p.pid, signal.SIGKILL)
        except Exception as e:
            if verbose:
                print(f"[!] Failed to terminate process PID {p.pid}: {e}")
            traceback.print_exc()

    # Wait briefly to allow processes to shut down
    time.sleep(0.5)

    if verbose:
        print("[✓] All processes attempted to terminate. Exiting.")
    os._exit(code)  # Bypass cleanup and exit immediately
