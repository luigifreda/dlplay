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
import time, datetime
from collections import defaultdict
import torch

from dlplay.core.smoothed_value import SmoothedValue


class MetricLogger:
    def __init__(self, delimiter="\t", writer=print):
        self.meters = defaultdict(
            SmoothedValue
        )  # assumes SmoothedValue(name, fmt=...) signature
        self.delimiter = delimiter
        self.writer = writer

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    raise ValueError(
                        f"Metric '{k}' must be a scalar; got shape {tuple(v.shape)}"
                    )
                v = v.item()
            if not isinstance(v, (float, int)):
                raise TypeError(f"Metric '{k}' must be float or int; got {type(v)}")
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {attr!r}")

    def __str__(self):
        return self.delimiter.join(f"{n}: {m}" for n, m in self.meters.items())

    def synchronize_between_processes(self):
        is_dist = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        for meter in self.meters.values():
            # Only call if the meter supports it and we’re actually distributed
            if is_dist and hasattr(meter, "synchronize_between_processes"):
                meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=""):
        if print_freq is None or print_freq <= 0:
            print_freq = 1
        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        has_len = hasattr(iterable, "__len__")
        total = len(iterable) if has_len else None
        space_fmt = ":" + str(len(str(total))) + "d" if has_len else ""

        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    f"[{{0{space_fmt}}}/{{1}}]" if has_len else "[{0}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    f"[{{0{space_fmt}}}/{{1}}]" if has_len else "[{0}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )

        MB = 1024.0 * 1024.0
        for i, obj in enumerate(iterable):
            now = time.time()
            data_time.update(now - end)
            yield i, obj
            iter_time.update(time.time() - now)

            should_print = i % print_freq == 0
            if has_len:
                should_print = should_print or (i == total - 1)
                remaining = max(total - i - 1, 0)
                eta_seconds = iter_time.global_avg * remaining
            else:
                eta_seconds = iter_time.global_avg * (print_freq - (i % print_freq) - 1)

            if should_print:
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    mem = torch.cuda.max_memory_allocated() / MB
                    self.writer(
                        log_msg.format(
                            i if not has_len else i + 1,
                            total,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=mem,
                        )
                    )
                else:
                    self.writer(
                        log_msg.format(
                            i if not has_len else i + 1,
                            total,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            end = time.time()

        elapsed = time.time() - start_time
        denom = total if total and total > 0 else (i + 1)
        self.writer(
            f"{header} Total time: {str(datetime.timedelta(seconds=int(elapsed)))} "
            f"({elapsed / denom:.4f} s / it)"
        )


if __name__ == "__main__":
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: 0"
    print_freq = 10

    # how to use it?
    num_iterations = 100
    lr = 0.0001
    loss = 1000
    for iter_idx, obj in metric_logger.log_every(
        range(num_iterations), print_freq, header
    ):
        loss *= 0.99
        lr *= 0.99
        time.sleep(0.05)  # simulate some processing time
        metric_logger.update(loss=loss, lr=lr)
