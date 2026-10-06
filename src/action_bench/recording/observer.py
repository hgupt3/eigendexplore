"""Fixed-epoch scheduling composed with each host's native training observer."""

import math
import os

DEFAULT_CAPTURE_WINDOW_SECONDS = 30.0


def make_capture_schedule(schedule, *, every_epochs=None):
    if every_epochs is not None:
        schedule = ({"every_epochs": every_epochs, "through_epoch": None},)
    previous = 0
    for index, entry in enumerate(schedule):
        every, through = entry["every_epochs"], entry["through_epoch"]
        if type(every) is not int or every < 1:
            raise ValueError("capture every_epochs must be a positive integer")
        if through is None:
            if index != len(schedule) - 1:
                raise ValueError("only the final capture period may be open-ended")
        elif type(through) is not int or through <= previous:
            raise ValueError("capture period boundaries must increase")
        previous = through
    return tuple(schedule)


def next_capture_due_epoch(schedule, epoch):
    start = 0
    for entry in schedule:
        every, through = entry["every_epochs"], entry["through_epoch"]
        candidate = start + (max(0, (epoch - start) // every) + 1) * every
        if through is None or candidate <= through:
            return candidate
        start = through
    return math.inf


class EpochCaptureScheduler:
    def __init__(self, schedule):
        self.schedule = make_capture_schedule(schedule)
        self.next_due_epoch = 0 if self.schedule else math.inf

    def poll(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if epoch < self.next_due_epoch:
            return False
        self.next_due_epoch = next_capture_due_epoch(self.schedule, epoch)
        return True


class StateCaptureObserver:
    """Delegate native callbacks and capture launch plus fixed training epochs."""

    def __init__(self, capture, schedule, observer):
        override = os.environ.get("ACTION_BENCH_RECORD_EVERY_EPOCHS")
        self.capture = capture
        self.schedule = make_capture_schedule(
            schedule, every_epochs=int(override) if override else None
        )
        self.observer = observer
        self.scheduler = EpochCaptureScheduler(self.schedule)

    def __getattr__(self, name):
        return getattr(self.observer, name)

    def after_init(self, algo):
        self.observer.after_init(algo)
        self.scheduler.poll(0)
        self.capture.arm(0, 0)

    def after_print_stats(self, frame, epoch_num, total_time):
        self.observer.after_print_stats(frame, epoch_num, total_time)
        self._capture_due(frame, epoch_num)

    def wandb_after_print_stats(self, frame, epoch_num, total_time):
        result = self.observer.wandb_after_print_stats(frame, epoch_num, total_time)
        self._capture_due(frame, epoch_num)
        return result

    def _capture_due(self, frame, epoch_num):
        self.capture.raise_if_failed()
        if self.scheduler.poll(int(epoch_num)):
            self.capture.arm(int(epoch_num), int(frame))
