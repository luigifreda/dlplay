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
import os
import sys
import time


class ProgressBar:

    def __init__(self, total_bar_length=65.0):
        self.term_width = int(os.popen("stty size", "r").read().split()[1])
        self.total_bar_length = total_bar_length
        self.last_time = time.time()
        self.begin_time = self.last_time

    def step(self, current, total, msg=None):
        if current == 0:
            self.begin_time = time.time()  # Reset for new bar.

        cur_len = int(self.total_bar_length * current / total)
        rest_len = int(self.total_bar_length - cur_len) - 1

        sys.stdout.write(" [")
        for i in range(cur_len):
            sys.stdout.write("=")
        sys.stdout.write(">")
        for i in range(rest_len):
            sys.stdout.write(".")
        sys.stdout.write("]")

        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        L = []
        L.append("  Step: %s" % self.format_time(step_time))
        L.append(" | Tot: %s" % self.format_time(tot_time))
        if msg:
            L.append(" | " + msg)

        msg = "".join(L)
        sys.stdout.write(msg)
        for i in range(self.term_width - int(self.total_bar_length) - len(msg) - 3):
            sys.stdout.write(" ")

        # Go back to the center of the bar.
        for i in range(self.term_width - int(self.total_bar_length / 2) + 2):
            sys.stdout.write("\b")
        sys.stdout.write(" %d/%d " % (current + 1, total))

        if current < total - 1:
            sys.stdout.write("\r")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def format_time(self, seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)

        f = ""
        i = 1
        if days > 0:
            f += str(days) + "D"
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + "h"
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + "m"
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + "s"
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + "ms"
            i += 1
        if f == "":
            f = "0ms"
        return f


if __name__ == "__main__":
    progress_bar = ProgressBar()
    for i in range(100):
        progress_bar.step(i, 100, "Testing")
        time.sleep(0.1)
