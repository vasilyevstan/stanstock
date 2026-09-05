from django.contrib import admin

from stanstock.core.admin_mixins import ReadOnlyModelAdmin
from stanstock.core.models import JobRun

admin.site.register(JobRun, ReadOnlyModelAdmin)
