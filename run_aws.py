from deepmanager import DeepManager


dm = DeepManager()

task = dm.get_task("test")

jobs = [task.create_job()]

with dm.require_instances("p2.xlarge", task.env, 1, terminate=True) as instances:
    instances.run_jobs(jobs)
