from django.urls import path

from . import views

urlpatterns = [
    path("", views.IndexView.as_view(), name="index"),
    path("repo/<int:pk>/", views.RepoDetailView.as_view(), name="repo-detail"),
]
