import re

from django import forms
from django.core.exceptions import ValidationError
from django.db.models import OuterRef, Subquery
from django.urls import reverse_lazy
from django.views.generic import CreateView, DetailView
from django.views.generic.edit import FormMixin
from django_tables2 import SingleTableView

from reporater.tasks import NamedTaskDatabaseBackend

from .models import Rating, Repo
from .tables import RepoTable
from .tasks import rate_repository


def extract_github_info(url: str) -> tuple[str, str] | tuple[None, None]:
    """
    Extract owner and repository name from a GitHub URL using regex with named groups.

    Args:
        url (str): GitHub URL in any common format

    Returns:
        tuple: (owner, repo_name)
    """
    # Pattern matches standard GitHub URLs with named capture groups
    pattern = r"github\.com\/(?P<owner>[\w.-]+)\/(?P<repo>[\w.-]+)"

    # Search for the pattern in the URL
    match = re.search(pattern, url)

    if match:
        # Extract owner and repo using the named groups
        owner = match.group("owner")
        repo = match.group("repo")
        return owner, repo

    return None, None


# Create a form for URL input
class RepoUrlForm(forms.Form):
    url = forms.URLField(label="Repository URL", widget=forms.URLInput(attrs={"placeholder": "Enter repository URL"}))

    def clean_url(self):
        """
        Custom validation to check if the URL already exists in the database.
        This method is automatically called during form validation.
        """
        url = self.cleaned_data["url"]

        owner, name = extract_github_info(url)
        if not owner or not name:
            raise ValidationError("This link is not a valid link to a Github repository.")

        return url


class IndexView(FormMixin, SingleTableView):
    template_name = "core/index.html"
    context_object_name = "repos"
    table_class = RepoTable
    form_class = RepoUrlForm
    success_url = reverse_lazy("index")  # Redirect to the same page after form submission

    def get_queryset(self):
        latest_rating = Rating.objects.filter(repo=OuterRef("pk")).order_by("-created_at")
        repos_queryset = Repo.objects.annotate(
            newest_rating=Subquery(latest_rating.values("rating")[:1]),
            last_rated_at=Subquery(latest_rating.values("created_at")[:1]),
        )

        return repos_queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if "form" not in context:
            context["form"] = self.get_form()
        return context

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form)
        else:
            self.object_list = []
            return self.form_invalid(form)

    def form_valid(self, form):
        url = form.cleaned_data["url"]
        owner, repo_name = extract_github_info(url)

        repo, _ = Repo.objects.get_or_create(url=url, owner=owner, name=repo_name)

        kwargs = {"repo_id": repo.id, "owner": owner, "repo": repo_name, NamedTaskDatabaseBackend.NAME_KEY: url}

        rate_repository.enqueue(**kwargs)

        return super().form_valid(form)


class RepoCreateView(CreateView):
    model = Repo
    template_name = "core/index.html"
    fields = ["repo", "owner", "url"]


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
