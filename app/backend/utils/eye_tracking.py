import clr  # Python for .NET — bridges CPython and the CLR
from time import time, sleep
from multiprocessing import Process, Value, Lock

# Add Tobii SDK assembly reference
clr.AddReference("../lib/dlls/tobii_interaction_lib_cs")
import Tobii.InteractionLib as tobii  # type: ignore


class EyeTracker:
    def __init__(self, save_path, resolution):
        self.process = None  # worker process handle
        self.running = Value("b", False)  # cross-process running flag
        self.lock = Lock()  # lock for thread-safe file writes
        self.width, self.height = resolution  # display resolution (pixels)
        self.rounding = 2  # decimal places for coordinates
        self.save_path = save_path  # path to the raw CSV output file

    def event_handler(self, event: tobii.GazePointData):
        """Callback invoked by the Tobii SDK for each gaze sample."""
        x = round(event.x, self.rounding)
        y = round(event.y, self.rounding)
        is_valid = event.validity == tobii.Validity.Valid
        if is_valid and x > 0 and y > 0:
            t = round(time() * 1000, self.rounding)  # timestamp in milliseconds
            data = "{},{},{}\n".format(t, x, y)
            with self.lock:
                with open(self.save_path, "a") as f:
                    f.write(data)
                    print("{},{},{}\n".format(t, x, y))

    def start_process(self):
        """Spawn a new process to begin eye-tracking data collection."""
        if not self.running.value:
            self.running.value = True
            self.process = Process(target=self.track)
            self.process.start()
            print("Eye tracking started.")

    def track(self):
        """Main loop running in the worker process; polls the Tobii SDK."""
        lib = tobii.InteractionLibFactory.CreateInteractionLib(
            tobii.FieldOfUse.Interactive
        )
        lib.CoordinateTransformAddOrUpdateDisplayArea(self.width, self.height)
        lib.CoordinateTransformSetOriginOffset(0, 0)
        lib.GazePointDataEvent += lambda event: self.event_handler(event)

        while self.running.value:
            lib.WaitAndUpdate()

    def stop_process(self):
        """Signal the worker process to stop and wait for it to exit."""
        if self.running and self.process is not None:
            self.running.value = False
            self.process.join()
            print("Eye tracking stopped.")

    def is_running(self):
        return self.running.value


if __name__ == "__main__":
    eye_tracker = EyeTracker()
    eye_tracker.start_process()
    sleep(60)
    eye_tracker.stop_process()
