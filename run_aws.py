from deepmanager import DeepManager


def main():
    dm = DeepManager()

    task = dm.get_task("test")

    jobs = [task.create_job()]

    with dm.require_instances("p2.xlarge", task.env, 1, terminate=False) as instances:
        instances.run_jobs(jobs)
