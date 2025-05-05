from django_tasks import Task
from django_tasks.backends.database import DatabaseBackend
from django_tasks.backends.database.backend import P, T, TaskResult
from django_tasks.backends.database.models import DBTaskResult


class TaskAlreadyRunningError(Exception):
    def __init__(self, task_name: str):
        self.task_name = task_name
        super().__init__(f"Task already running: {task_name}")


class NamedTaskDatabaseBackend(DatabaseBackend):
    NAME_KEY = "NamedTaskDatabaseBackend_META_taskname"

    def enqueue(
        self,
        task: Task[P, T],
        args: P.args,  # type:ignore[valid-type]
        kwargs: P.kwargs,  # type:ignore[valid-type]
    ) -> TaskResult[T]:
        name = kwargs.get(self.NAME_KEY)
        if not name:
            msg = f"A keyword-argument named '{self.NAME_KEY}' is required"
            raise ValueError(msg)
        elif (
            DBTaskResult.objects.filter(**{f"args_kwargs__kwargs__{self.NAME_KEY}": name})
            .filter(finished_at__isnull=True)
            .exists()
        ):
            raise TaskAlreadyRunningError(name)

        return super().enqueue(task, args, kwargs)
