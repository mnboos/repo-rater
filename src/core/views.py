from django.db.models import OuterRef, Subquery
from django.views.generic import DetailView
from django_tables2 import SingleTableView

from .models import Rating, Repo
from .tables import RepoTable


class IndexView(SingleTableView):
    template_name = "core/index.html"
    context_object_name = "repos"
    table_class = RepoTable

    def get_queryset(self):
        latest_rating = Rating.objects.filter(repo=OuterRef("pk")).order_by("-created_at")
        repos_queryset = Repo.objects.annotate(
            newest_rating=Subquery(latest_rating.values("rating")[:1]),
            last_rated_at=Subquery(latest_rating.values("created_at")[:1]),
        )

        return repos_queryset


class RepoDetailView(DetailView):
    model = Repo
    template_name = "core/repo_detail.html"
    context_object_name = "repo"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Get the latest rating
        latest_rating = self.object.ratings.order_by("-created_at").first()
        context["latest_rating"] = latest_rating

        # Get all ratings for the repo
        context["ratings"] = self.object.ratings.order_by("-created_at")
        return context
