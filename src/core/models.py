from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    pass


class Repo(models.Model):
    class Meta:
        unique_together = [("owner", "name")]

    owner = models.CharField()
    name = models.CharField()
    url = models.URLField(unique=True)

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        from django.urls import reverse

        return reverse("repo-detail", kwargs={"pk": self.pk})


# Create your models here.
class Rating(models.Model):
    repo = models.ForeignKey(Repo, related_name="ratings", on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    rating = models.FloatField()
