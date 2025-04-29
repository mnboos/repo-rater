from django.contrib import admin

# Register your models here.

from .models import Rating, User, Repo


class RatingInline(admin.TabularInline):
    """Inline für Bewertungskriterien unter Faehigkeitsdimension."""

    model = Rating
    extra = 0
    fields = (
        "rating",
        "created_at",
    )
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)


admin.site.register(User)


@admin.register(Repo)
class RepoAdmin(admin.ModelAdmin):
    list_display = ("name", "url")
    search_fields = ("name",)
    inlines = [RatingInline]
