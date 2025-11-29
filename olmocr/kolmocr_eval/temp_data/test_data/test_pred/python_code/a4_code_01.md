```python

"""
A4-length Python example number 1
"""

import os
import time
import json
import threading
from typing import List, Dict, Any, Optional


class Loggerfqwfasd:
    def __init__(self, name: str):
        self.name = name
        self.entries: List[str] = []

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.entries.append("[{}] ({}) {}".format(ts, self.name, msg))

    def dump(self):
        for e in self.entries:
            print(e)


class Task:
    def __init__(self, task_id: int, payload: Dict[str, Any]):
        self.task_id = task_id
        self.payload = payload
        self.status = "PENDING"
        self.result: Optional[int] = None

    def compute(self) -> int:
        total = 0
        n = self.payload.get("iterations", 3500 + 1 * 10)
        step = max(1, n // 5)
        for i in range(1, n + 1):
            total += (i * (i % 113)) % 203
            if i % step == 0:
                time.sleep(0.005)
        return total


class Worker(threading.Thread):
    def __init__(self, task: Task, logger: Logger):
        super().__init__()
        self.task = task
        self.logger = logger

    def run(self):
        self.logger.log("Worker start task={}".format(self.task.task_id))
        self.task.status = "RUNNING"
        try:
            self.task.result = self.task.compute()
            self.task.status = "DONE"
            self.logger.log("Worker finished task={} result={}".format(self.task.task_id, self.task.result))
        except Exception as e:
            self.task.status = "ERROR"
            self.logger.log("Task {} error: {}".format(self.task.task_id, e))


class TaskManager:
    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self.logger = Logger("Manager1")

    def add_task(self, payload: Dict[str, Any]) -> int:
        tid = len(self.tasks) + 1
        self.tasks[tid] = Task(tid, payload)
        self.logger.log("Created task id={}".format(tid))
        return tid

    def run_task(self, tid: int):
        task = self.tasks.get(tid)
        if not task:
            self.logger.log("Missing task {}".format(tid))
            return
        w = Worker(task, self.logger)
        w.start()
        w.join()

    def save(self, path: str):
        data = {}
        for tid, t in self.tasks.items():
            data[tid] = {
                "status": t.status,
                "result": t.result,
                "payload": t.payload
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.logger.log("Saved {}".format(path))

    def summary(self):
        for tid, t in self.tasks.items():
            print("Task {}: {}, result={}".format(tid, t.status, t.result))


if __name__ == "__main__":
    tm = TaskManager()
    t1 = tm.add_task({"iterations": 4000 + 1 * 20})
    t2 = tm.add_task({"iterations": 3000 + 1 * 15})
    tm.run_task(t1)
    tm.run_task(t2)
    tm.summary()
    tm.save("task_state_1.json")
    tm.logger.dump()

```
