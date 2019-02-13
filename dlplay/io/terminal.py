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

import sys
import os

### Color printing functions from pySLAM

# ANSI escape codes for text colors
TERMINAL_COLORS = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "orange": "\033[33m",  # ANSI does not have orange, using yellow
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bold": "\033[1m",
}


def text_color(text, color):
    """Return text wrapped in ANSI color codes."""
    return f"{TERMINAL_COLORS.get(color, TERMINAL_COLORS['reset'])}{text}{TERMINAL_COLORS['reset']}"


def print_red(text):
    print(text_color(text, "red"))


def print_green(text):
    print(text_color(text, "green"))


def print_yellow(text):
    print(text_color(text, "yellow"))


def print_blue(text):
    print(text_color(text, "blue"))


def print_orange(text):
    print(text_color(text, "yellow"))


def print_magenta(text):
    print(text_color(text, "magenta"))


def print_cyan(text):
    print(text_color(text, "cyan"))


def print_bold(text):
    print(f"{TERMINAL_COLORS['bold']}{text}{TERMINAL_COLORS['reset']}")


def get_char_to_continue():
    print_blue("Press enter to continue/exit...")
    a = input("").split(" ")[0]
    print(a)
