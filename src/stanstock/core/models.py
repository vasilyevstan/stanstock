from __future__ import annotations

import uuid

from django.db import models


class JobRun(models.Model):
    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"
        NO_DATA = "no_data", "No data"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job_name = models.CharField(max_length=80)
    region = models.CharField(max_length=20, blank=True)
    target_date = models.DateField()
    attempt = models.PositiveSmallIntegerField(default=1)
    status = models.CharField(max_length=16, choices=Status, default=Status.RUNNING)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    details = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["job_name", "region", "target_date", "attempt"],
                name="unique_job_target_attempt",
            ),
            models.UniqueConstraint(
                fields=["job_name", "region", "target_date"],
                condition=models.Q(status="success"),
                name="unique_successful_job_target",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(status="running", finished_at__isnull=True)
                    | (~models.Q(status="running") & models.Q(finished_at__isnull=False))
                ),
                name="job_finished_at_matches_status",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.job_name}:{self.region}:{self.target_date}:{self.status}"
