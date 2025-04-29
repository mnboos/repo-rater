import django_tables2 as tables

from .models import Repo


class RepoTable(tables.Table):
    class Meta:
        model = Repo
        exclude = ("id",)
        sequence = ("name", "rating", "rated_at", "url")

    name = tables.Column(accessor="name", linkify=True)
    rating = tables.Column(accessor="newest_rating", verbose_name="Rating")
    rated_at = tables.Column(accessor="last_rated_at", verbose_name="Rated at", orderable=True)
    url = tables.Column(accessor="url", orderable=False)
