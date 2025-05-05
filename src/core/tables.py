import django_tables2 as tables

from .models import Repo


class NumberColumn(tables.Column):
    def render(self, value):
        return "{:0.2f}".format(value)


class PercentColumn(tables.Column):
    def render(self, value):
        return f"{value:.0%}"


class RepoTable(tables.Table):
    class Meta:
        model = Repo
        exclude = ("id", "owner")
        sequence = ("name", "rating", "rated_at", "url")

    name = tables.Column(linkify=True)
    rating = PercentColumn(accessor="newest_rating", verbose_name="Rating")
    rated_at = tables.DateTimeColumn(accessor="last_rated_at", verbose_name="Rated at", orderable=True)
    url = tables.URLColumn(verbose_name="Link", orderable=False, attrs={"a": {"target": "_blank"}})
