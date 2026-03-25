# src/utils/logger.py

import time
import json
import os
import numpy as np


class PlanningLogger:
    """Logs planning metrics across multiple runs."""

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.runs = []
        self._collision_count = 0
        self._start_time = None

    def start_run(self, density, num_objects, run_id):
        """Call before each planning attempt."""
        self._collision_count = 0
        self._start_time = time.time()
        self._current_run = {
            "density": density,
            "num_objects": num_objects,
            "run_id": run_id,
        }

    def log_collision_check(self):
        """Call every time in_collision is invoked."""
        self._collision_count += 1

    def end_run(self, path, success):
        """Call after planning finishes."""
        elapsed = time.time() - self._start_time

        path_length = 0.0
        if success and path is not None and len(path) > 1:
            for i in range(1, len(path)):
                path_length += np.linalg.norm(
                    np.array(path[i]) - np.array(path[i - 1])
                )

        self._current_run.update({
            "success": success,
            "planning_time_s": round(elapsed, 4),
            "path_length": round(path_length, 4),
            "num_waypoints": len(path) if path else 0,
            "collision_checks": self._collision_count,
        })

        self.runs.append(self._current_run)
        self._current_run = None

    def save(self, filename="planning_log.json"):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.runs, f, indent=2)
        print(f"Log saved to {filepath}")

    def get_runs(self):
        return self.runs