"""
GitHub Repository Quality Score (GRQS) - A tool to evaluate GitHub repositories
based on multiple quantitative and qualitative metrics. (Automated Version)
"""

import argparse
import dataclasses
import datetime
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from getpass import getpass
from typing import Any, Dict, List, Optional, Tuple, Union

import django_tasks
import requests
from dateutil.parser import parse

from core.models import Rating

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("github_repo_rater")


@dataclass
class ActivityData:
    commit_frequency_score: float
    recent_activity_score: float
    release_frequency_score: float
    days_since_last_commit: Optional[int]
    num_commits_6months: int
    num_releases_6months: int


@dataclass
class MaintenanceData:
    avg_issue_open_time_score: float  # Replaces response time
    pr_merge_rate_score: float
    update_cadence_score: float  # Based on repo updated_at
    num_open_issues: int
    num_closed_prs_recent: int
    days_since_last_update: int


@dataclass
class DocumentationData:
    readme_score: float
    contribution_guidelines_score: float  # CONTRIBUTING.md, CODE_OF_CONDUCT.md
    examples_score: float  # Check for examples/samples dir
    wiki_score: float  # Based on has_wiki flag
    has_readme: bool
    readme_size: Optional[int]
    has_contributing: bool
    has_code_of_conduct: bool
    has_examples: bool
    has_wiki: bool


@dataclass
class CommunityData:
    contributor_count: int
    contributor_growth_score: float
    issue_activity_score: float  # Based on open/closed ratio
    community_health_score: float  # Based on CONTRIB/CoC files
    open_issues: int
    total_issues_approx: int  # Approximation


@dataclass
class ComponentData:
    score: float
    weight: float
    data: Union[Dict[str, Any], ActivityData, MaintenanceData, DocumentationData, CommunityData]


@dataclass
class ResultData:
    repo: str
    final_score: float
    rating: str
    components: Dict[str, ComponentData]


class GitHubAPIClient:
    """Client for interacting with the GitHub API."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the GitHub API client with an optional token."""
        self.token = token
        self.headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    def make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Make a request to the GitHub API with rate limit handling."""
        try:
            response = requests.get(url, headers=self.headers, params=params)

            # Check for rate limiting
            if response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                logger.debug(f"Rate limit remaining: {remaining}")
                if remaining == 0:
                    reset_time = int(response.headers["X-RateLimit-Reset"])
                    sleep_time = max(0, reset_time - time.time()) + 1  # Add buffer
                    logger.warning(f"Rate limit hit. Waiting for {sleep_time:.0f} seconds...")
                    time.sleep(sleep_time)
                    return self.make_request(url, params)  # Retry

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Specifically handle 404 Not Found without logging error unless debug
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                logger.debug(f"Resource not found (404): {url}")
                return None
            logger.error(f"Error making request to {url}: {e}")
            return None

    def get_paginated_results(
        self, url: str, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> List[Any]:
        """Get all results from a paginated API endpoint."""
        if params is None:
            params = {}

        results: List[Any] = []
        page = 1
        max_per_page = 100  # Max allowed by GitHub API

        while True:
            current_params = params.copy()
            current_params["page"] = page
            current_params["per_page"] = max_per_page

            page_results = self.make_request(url, current_params)

            # Handle potential None return from make_request after error/404
            if page_results is None:
                logger.warning(f"Received no results for page {page} of {url}. Stopping pagination.")
                break
            # Handle cases where the API returns a dict instead of a list (e.g., error message)
            if not isinstance(page_results, list):
                logger.warning(
                    f"Expected list but got {type(page_results)} for page {page} of {url}. Stopping pagination."
                )
                break
            if not page_results:  # Empty list means no more pages
                break

            results.extend(page_results)

            # Stop if we fetched less than requested per page (must be the last page)
            if len(page_results) < max_per_page:
                break

            # Stop if we reached a specified limit
            if limit and len(results) >= limit:
                results = results[:limit]
                break

            page += 1
            # Optional: Add a small delay between pages to be polite to the API
            time.sleep(0.1)

        return results

    def check_path_exists(self, owner: str, repo: str, path: str) -> bool:
        """Check if a file or directory exists at the given path."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        # We only need headers, so use requests.head if possible, but GET works fine too
        # response = requests.head(url, headers=self.headers)
        # return response.status_code == 200
        # Using GET via make_request to handle rate limits etc.
        response_data = self.make_request(url)
        return response_data is not None


# --- Data Collector ---
class DataCollector:
    """Collects required data from GitHub for repository rating."""

    def __init__(self, api_client: GitHubAPIClient) -> None:
        """Initialize with a GitHub API client."""
        self.api_client = api_client

    def get_repo_info(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get basic repository information."""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        return self.api_client.make_request(url)

    def get_commits(
        self,
        owner: str,
        repo: str,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get commits for a repository with optional date filter."""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params: Dict[str, Any] = {}
        if since_date:
            params["since"] = since_date
        if until_date:
            params["until"] = until_date

        # Use get_paginated_results which handles pagination and errors
        return self.api_client.get_paginated_results(url, params, limit)

    def get_releases(self, owner: str, repo: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get releases for a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        # Releases are often fewer, but pagination is good practice
        return self.api_client.get_paginated_results(url, limit=limit)

    def get_issues(
        self, owner: str, repo: str, state: str = "all", since_date: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get issues for a repository (includes PRs by default)."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params: Dict[str, Any] = {"state": state, "sort": "updated", "direction": "desc"}
        if since_date:
            params["since"] = since_date  # Issues updated since this time

        results = self.api_client.get_paginated_results(url, params, limit)
        # Filter out pull requests if only issues are needed (more robust)
        # return [issue for issue in results if 'pull_request' not in issue]
        return results  # Keep PRs for now, filter later if needed

    def get_pulls(self, owner: str, repo: str, state: str = "closed", limit: int = 100) -> List[Dict[str, Any]]:
        """Get pull requests for a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {"state": state, "sort": "updated", "direction": "desc"}
        return self.api_client.get_paginated_results(url, params, limit)

    def get_contributors(self, owner: str, repo: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get contributors for a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
        # Contributors can be numerous, paginate
        return self.api_client.get_paginated_results(url, params={"anon": "true"}, limit=limit)  # include anonymous

    def get_readme(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get readme information (including size)."""
        url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        return self.api_client.make_request(url)  # Returns metadata including size

    def check_file_exists(self, owner: str, repo: str, file_path: str) -> bool:
        """Check if a specific file exists."""
        return self.api_client.check_path_exists(owner, repo, file_path)

    def check_dir_exists(self, owner: str, repo: str, dir_path: str) -> bool:
        """Check if a directory exists (API returns array for dirs)."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}"
        response = self.api_client.make_request(url)
        # Check if response is a list (indicating a directory)
        return isinstance(response, list)


class MetricsCalculator:
    """Calculates individual metrics scores (0.0 - 1.0) for repository rating."""

    def __init__(self):
        self.now = datetime.datetime.now(datetime.timezone.utc)

    @staticmethod
    def calculate_stars_score(star_count: int) -> float:
        """Calculate score based on star count using logarithmic scale."""
        # Normalize based on log10, aiming for 1.0 around 100k stars
        return min(math.log10(max(star_count, 1)) / 5.0, 1.0)

    @staticmethod
    def calculate_commit_frequency_score(commits: List[Dict[str, Any]], months_period: int) -> float:
        """Calculate commit frequency score (normalized)."""
        if months_period <= 0:
            return 0.0
        total_commits = len(commits)
        # Normalize based on commits per month, aiming for 1.0 around 50 commits/month (adjustable)
        commits_per_month = total_commits / months_period
        return min(commits_per_month / 50.0, 1.0)

    def calculate_recent_activity_score(self, last_commit_date_str: Optional[str]) -> Tuple[float, Optional[int]]:
        """Calculate recent activity score based on days since last commit."""
        if not last_commit_date_str:
            return 0.0, None
        try:
            last_commit_date = parse(last_commit_date_str)
            days_since = (self.now - last_commit_date).days
            # Exponential decay, score halves roughly every 30 days
            score = math.exp(-days_since / 43.28)  # ln(2) / 30 days ~= 0.0231 -> 1/0.0231 ~= 43.28
            return max(0.0, min(score, 1.0)), days_since
        except Exception as e:
            logger.warning(f"Could not parse last commit date '{last_commit_date_str}': {e}")
            return 0.0, None

    def calculate_release_frequency_score(
        self, releases: List[Dict[str, Any]], months_period: int
    ) -> Tuple[float, int]:
        """Calculate release frequency score."""
        if months_period <= 0 or not releases:
            return 0.0, 0
        cutoff_date = self.now - datetime.timedelta(days=30 * months_period)
        recent_releases = [r for r in releases if parse(r["published_at"]) > cutoff_date]
        num_recent_releases = len(recent_releases)
        # Normalize based on releases per month, aiming for 1.0 around 1 release/month
        releases_per_month = num_recent_releases / months_period
        return min(releases_per_month / 1.0, 1.0), num_recent_releases

    def calculate_avg_issue_open_time_score(self, open_issues: List[Dict[str, Any]]) -> float:
        """Calculate score based on average time issues have been open."""
        if not open_issues:
            return 1.0  # No open issues is good

        total_open_days = 0
        valid_issues = 0
        for issue in open_issues:
            if "pull_request" in issue:
                continue  # Skip PRs listed as issues
            try:
                created_at = parse(issue["created_at"])
                open_days = (self.now - created_at).days
                total_open_days += open_days
                valid_issues += 1
            except Exception as e:
                logger.debug(f"Could not parse issue created_at date: {e}")
                continue

        if valid_issues == 0:
            return 0.75  # No valid issues to calculate from, neutral score

        avg_open_days = total_open_days / valid_issues
        # Score decreases as average open time increases. Halves around 60 days open.
        score = math.exp(-avg_open_days / 86.56)  # ln(2) / 60 days ~= 0.01155 -> 1/0.01155 ~= 86.56
        return max(0.0, min(score, 1.0))

    @staticmethod
    def calculate_pr_merge_rate_score(closed_prs: List[Dict[str, Any]]) -> Tuple[float, int]:
        """Calculate pull request merge rate score."""
        if not closed_prs:
            return 0.5, 0  # Neutral score if no closed PRs found recently

        merged_prs = sum(1 for pr in closed_prs if pr.get("merged_at"))
        total_closed = len(closed_prs)
        merge_rate = merged_prs / total_closed
        # Directly use the merge rate as the score
        return max(0.0, min(merge_rate, 1.0)), total_closed

    def calculate_update_cadence_score(self, repo_updated_at_str: str) -> Tuple[float, int]:
        """Calculate score based on how recently the repo was updated."""
        try:
            repo_updated_at = parse(repo_updated_at_str)
            days_since_update = (self.now - repo_updated_at).days
            # Similar decay to recent commit score, maybe slightly slower
            score = math.exp(-days_since_update / 60.0)  # Halves around 42 days
            return max(0.0, min(score, 1.0)), days_since_update
        except Exception as e:
            logger.warning(f"Could not parse repo updated_at date '{repo_updated_at_str}': {e}")
            return 0.0, 9999  # Penalize if date is unparsable

    @staticmethod
    def calculate_readme_score(has_readme: bool, readme_size: Optional[int]) -> float:
        """Calculate README score based on existence and size."""
        if not has_readme:
            return 0.0
        # Base score for existence
        score = 0.5
        if readme_size is not None:
            # Add bonus for size (log scale), max bonus 0.5 for ~10KB+
            size_bonus = min(math.log10(max(readme_size, 1)) / 4.0, 0.5)  # log10(10000) = 4
            score += size_bonus
        return min(score, 1.0)

    @staticmethod
    def calculate_contribution_guidelines_score(has_contributing: bool, has_code_of_conduct: bool) -> float:
        """Score based on presence of CONTRIBUTING and CODE_OF_CONDUCT files."""
        score = 0.0
        if has_contributing:
            score += 0.6
        if has_code_of_conduct:
            score += 0.4
        return score

    @staticmethod
    def calculate_examples_score(has_examples: bool) -> float:
        """Score based on presence of an examples directory."""
        return 1.0 if has_examples else 0.0

    @staticmethod
    def calculate_wiki_score(has_wiki: bool) -> float:
        """Score based on whether the repo has an enabled wiki."""
        return 0.8 if has_wiki else 0.2  # Bonus for having it, small penalty for not

    @staticmethod
    def calculate_contributor_growth_score(contributor_count: int) -> float:
        """Calculate score based on number of contributors (log scale)."""
        # Normalize log10, aiming for 1.0 around 1000 contributors
        return min(math.log10(max(contributor_count, 1)) / 3.0, 1.0)

    @staticmethod
    def calculate_issue_activity_score(open_issues_count: int, total_issues_approx: int) -> float:
        """Calculate score based on the ratio of open to total issues."""
        if total_issues_approx <= 0:
            return 0.5  # Neutral score if no issues

        open_ratio = open_issues_count / total_issues_approx
        # Score is 1 - open_ratio, penalizing repos with high proportion of open issues
        # Cap penalty, e.g., lowest score is 0.1 even if 100% open
        return max(0.1, 1.0 - open_ratio)

    @staticmethod
    def calculate_community_health_score(has_contributing: bool, has_code_of_conduct: bool) -> float:
        """Score based on presence of community health files (same as contribution guidelines)."""
        # This metric might be redundant with the documentation one, or weighted differently.
        # Using the same calculation for now.
        score = 0.0
        if has_contributing:
            score += 0.5
        if has_code_of_conduct:
            score += 0.5
        return score


# --- Component Calculator ---
class ComponentCalculator:
    """Calculates component scores (aggregating individual metrics)."""

    def __init__(self, data_collector: DataCollector, metrics_calculator: MetricsCalculator) -> None:
        """Initialize with data collector and metrics calculator."""
        self.data_collector = data_collector
        self.metrics = metrics_calculator
        self.now = datetime.datetime.now(datetime.timezone.utc)
        self.six_months_ago_dt = self.now - datetime.timedelta(days=180)
        self.six_months_ago_str = self.six_months_ago_dt.isoformat()

    def calculate_stars(self, repo_data: Dict[str, Any]) -> ComponentData:
        """Calculate Stars component score."""
        star_count = repo_data.get("stargazers_count", 0)
        score = self.metrics.calculate_stars_score(star_count)
        logger.debug(f"Stars: Count={star_count} -> Score={score:.2f}")
        return ComponentData(score=score, weight=0, data={"count": star_count})  # Weight applied later

    def calculate_activity(self, owner: str, repo: str) -> ComponentData:
        """Calculate Activity component score."""
        # Get commits from last 6 months (limit to avoid excessive data)
        commits = self.data_collector.get_commits(
            owner,
            repo,
            since_date=self.six_months_ago_str,
            limit=500,  # Limit to recent 500
        )
        num_commits_6m = len(commits) if commits else 0

        commit_frequency_score = self.metrics.calculate_commit_frequency_score(commits or [], 6)

        # Recent activity based on latest commit in the fetched list
        latest_commit_date_str = None
        if commits and len(commits) > 0:
            # Assume commits are sorted newest first by API default
            latest_commit_date_str = commits[0].get("commit", {}).get("committer", {}).get("date")

        recent_activity_score, days_since_last_commit = self.metrics.calculate_recent_activity_score(
            latest_commit_date_str
        )

        # Get releases (limit to recent ones for frequency)
        releases = self.data_collector.get_releases(owner, repo, limit=100) or []
        release_frequency_score, num_releases_6m = self.metrics.calculate_release_frequency_score(releases, 6)

        # Calculate overall activity score (adjust weights as needed)
        activity_score = 0.4 * commit_frequency_score + 0.4 * recent_activity_score + 0.2 * release_frequency_score

        logger.debug(
            f"Activity: Commits(6m)={num_commits_6m}, Releases(6m)={num_releases_6m}, DaysSinceCommit={days_since_last_commit}"
        )
        logger.debug(
            f"Activity Scores: CF={commit_frequency_score:.2f}, RA={recent_activity_score:.2f}, RF={release_frequency_score:.2f} -> Overall: {activity_score:.2f}"
        )

        activity_data = ActivityData(
            commit_frequency_score=commit_frequency_score,
            recent_activity_score=recent_activity_score,
            release_frequency_score=release_frequency_score,
            days_since_last_commit=days_since_last_commit,
            num_commits_6months=num_commits_6m,
            num_releases_6months=num_releases_6m,
        )
        return ComponentData(score=activity_score, weight=0, data=activity_data)

    def calculate_maintenance(self, owner: str, repo: str, repo_data: Dict[str, Any]) -> ComponentData:
        """Calculate Maintenance component score."""
        # Get recent issues (open state) to gauge average open time
        # Limit to avoid too much data, sort by creation time? or updated? updated is default
        open_issues_raw = self.data_collector.get_issues(owner, repo, state="open", limit=100)
        open_issues = [i for i in open_issues_raw if "pull_request" not in i]  # Filter out PRs
        num_open_issues = len(open_issues)

        avg_issue_open_time_score = self.metrics.calculate_avg_issue_open_time_score(open_issues)

        # Get recently closed pull requests (last 100)
        closed_prs = self.data_collector.get_pulls(owner, repo, state="closed", limit=100)
        pr_merge_rate_score, num_closed_prs = self.metrics.calculate_pr_merge_rate_score(closed_prs or [])

        # Update cadence based on repo's last update timestamp
        repo_updated_at = repo_data.get("updated_at")
        update_cadence_score, days_since_last_update = (
            self.metrics.calculate_update_cadence_score(repo_updated_at) if repo_updated_at else (0.0, 9999)
        )

        # Calculate overall maintenance score (adjust weights as needed)
        maintenance_score = 0.35 * avg_issue_open_time_score + 0.35 * pr_merge_rate_score + 0.30 * update_cadence_score

        logger.debug(
            f"Maintenance: OpenIssues={num_open_issues}, ClosedPRsChecked={num_closed_prs}, DaysSinceUpdate={days_since_last_update}"
        )
        logger.debug(
            f"Maintenance Scores: IssueTime={avg_issue_open_time_score:.2f}, PRMerge={pr_merge_rate_score:.2f}, UpdateCad={update_cadence_score:.2f} -> Overall: {maintenance_score:.2f}"
        )

        maintenance_data = MaintenanceData(
            avg_issue_open_time_score=avg_issue_open_time_score,
            pr_merge_rate_score=pr_merge_rate_score,
            update_cadence_score=update_cadence_score,
            num_open_issues=num_open_issues,
            num_closed_prs_recent=num_closed_prs,
            days_since_last_update=days_since_last_update,
        )
        return ComponentData(score=maintenance_score, weight=0, data=maintenance_data)

    def calculate_documentation(self, owner: str, repo: str, repo_data: Dict[str, Any]) -> ComponentData:
        """Calculate Documentation component score automatically."""
        # Check for README
        readme_info = self.data_collector.get_readme(owner, repo)
        has_readme = readme_info is not None
        readme_size = readme_info.get("size") if readme_info else None
        readme_score = self.metrics.calculate_readme_score(has_readme, readme_size)

        # Check for contribution files
        has_contributing = self.data_collector.check_file_exists(owner, repo, "CONTRIBUTING.md")
        has_code_of_conduct = self.data_collector.check_file_exists(owner, repo, "CODE_OF_CONDUCT.md")
        contribution_guidelines_score = self.metrics.calculate_contribution_guidelines_score(
            has_contributing, has_code_of_conduct
        )

        # Check for examples directory
        has_examples = (
            self.data_collector.check_dir_exists(owner, repo, "examples")
            or self.data_collector.check_dir_exists(owner, repo, "samples")
            or self.data_collector.check_dir_exists(owner, repo, "docs/examples")
        )  # Check common paths
        examples_score = self.metrics.calculate_examples_score(has_examples)

        # Check for wiki
        has_wiki = repo_data.get("has_wiki", False)
        wiki_score = self.metrics.calculate_wiki_score(has_wiki)

        # Calculate overall documentation score (adjust weights as needed)
        documentation_score = (
            0.4 * readme_score + 0.3 * contribution_guidelines_score + 0.15 * examples_score + 0.15 * wiki_score
        )

        logger.debug(
            f"Documentation Checks: README={has_readme}, CONTRIB={has_contributing}, CoC={has_code_of_conduct}, Examples={has_examples}, Wiki={has_wiki}"
        )
        logger.debug(
            f"Documentation Scores: Readme={readme_score:.2f}, Guidelines={contribution_guidelines_score:.2f}, Examples={examples_score:.2f}, Wiki={wiki_score:.2f} -> Overall: {documentation_score:.2f}"
        )

        documentation_data = DocumentationData(
            readme_score=readme_score,
            contribution_guidelines_score=contribution_guidelines_score,
            examples_score=examples_score,
            wiki_score=wiki_score,
            has_readme=has_readme,
            readme_size=readme_size,
            has_contributing=has_contributing,
            has_code_of_conduct=has_code_of_conduct,
            has_examples=has_examples,
            has_wiki=has_wiki,
        )
        return ComponentData(score=documentation_score, weight=0, data=documentation_data)

    def calculate_community(self, owner: str, repo: str, repo_data: Dict[str, Any]) -> ComponentData:
        """Calculate Community component score automatically."""
        # Get contributors (limit if potentially very large)
        contributors = self.data_collector.get_contributors(owner, repo, limit=1000)
        contributor_count = len(contributors) if contributors else 0
        contributor_growth_score = self.metrics.calculate_contributor_growth_score(contributor_count)

        # Issue activity score (based on open issues vs. repo lifetime/total issues)
        open_issues_count = repo_data.get("open_issues_count", 0)
        # Estimate total issues: Use open_issues + heuristic based on repo age/activity?
        # For simplicity, let's use a rough approximation or fetch closed issues if needed.
        # Simple approx: assume closed issues = open issues * factor (e.g., 2) or use total count if available.
        # GitHub API doesn't easily give total issue count (open+closed) directly on repo object.
        # Let's fetch recent closed issues to get a better estimate.
        closed_issues_raw = self.data_collector.get_issues(
            owner, repo, state="closed", limit=200
        )  # Get recent closed issues
        closed_issues_count = len([i for i in closed_issues_raw if "pull_request" not in i])
        total_issues_approx = open_issues_count + closed_issues_count
        # Fallback if no closed issues found
        if total_issues_approx == open_issues_count and open_issues_count > 0:
            total_issues_approx = open_issues_count * 2  # Simple guess if no closed issues found

        issue_activity_score = self.metrics.calculate_issue_activity_score(open_issues_count, total_issues_approx)

        # Community health score (based on guidelines files)
        # Reuse checks from documentation calculation if already done, or re-check
        has_contributing = getattr(
            self, "_has_contributing", self.data_collector.check_file_exists(owner, repo, "CONTRIBUTING.md")
        )
        has_code_of_conduct = getattr(
            self, "_has_code_of_conduct", self.data_collector.check_file_exists(owner, repo, "CODE_OF_CONDUCT.md")
        )
        # Cache these checks if needed across components
        self._has_contributing = has_contributing
        self._has_code_of_conduct = has_code_of_conduct
        community_health_score = self.metrics.calculate_community_health_score(has_contributing, has_code_of_conduct)

        # Calculate overall community score (adjust weights as needed)
        community_score = 0.4 * contributor_growth_score + 0.3 * issue_activity_score + 0.3 * community_health_score

        logger.debug(
            f"Community: Contribs={contributor_count}, OpenIssues={open_issues_count}, TotalIssuesEst={total_issues_approx}"
        )
        logger.debug(
            f"Community Scores: Growth={contributor_growth_score:.2f}, IssueActivity={issue_activity_score:.2f}, HealthFiles={community_health_score:.2f} -> Overall: {community_score:.2f}"
        )

        community_data = CommunityData(
            contributor_count=contributor_count,
            contributor_growth_score=contributor_growth_score,
            issue_activity_score=issue_activity_score,
            community_health_score=community_health_score,  # Replaces discussion quality
            open_issues=open_issues_count,
            total_issues_approx=total_issues_approx,
        )
        return ComponentData(score=community_score, weight=0, data=community_data)


# --- Input Provider (Simplified) ---
class InputProvider:
    """Handles user input for non-metric configuration like token and repo."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize with command line arguments."""
        self.args = args

    def get_github_token(self) -> Optional[str]:
        """Get GitHub token from args, environment, or prompt."""
        # 1. Command line argument
        if self.args.token:
            logger.debug("Using token from command line argument.")
            return self.args.token

        # 2. Environment variable
        env_token = os.environ.get("GITHUB_TOKEN")
        if env_token:
            logger.debug("Found token in GITHUB_TOKEN environment variable.")
            # Optionally prompt to confirm use, or just use it directly for automation
            # For full automation, we'll just use it if found.
            # use_env_token = input(f"Use GitHub token from environment? (y/n) [y]: ").lower() != 'n'
            # if use_env_token:
            #     return env_token
            return env_token  # Use directly

        # 3. Prompt user (optional, can be disabled for full automation)
        use_token_prompt = getattr(self.args, "prompt_token", False)  # Add --prompt-token flag if needed
        if use_token_prompt:
            use_token = input("No token found in args or env. Use a GitHub token? (y/n) [n]: ").lower() == "y"
            if use_token:
                return getpass("Enter GitHub token: ")
        else:
            logger.warning("No GitHub token provided via args or environment. API rate limits will be stricter.")

        return None

    # get_repo_info is removed as owner/repo are now mandatory args


# --- Repo Rater ---
class RepoRater:
    """Main class for rating GitHub repositories."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the repository rater with dependencies."""
        self.api_client = GitHubAPIClient(token)
        self.data_collector = DataCollector(self.api_client)
        self.metrics = MetricsCalculator()
        self.component_calculator = ComponentCalculator(self.data_collector, self.metrics)
        # InputProvider is no longer needed here for metrics

        # Component weights (Adjust these based on the perceived importance of each automated component)
        self.weights: Dict[str, float] = {
            "stars": 0.15,  # Popularity indicator
            "activity": 0.25,  # Development momentum
            "maintenance": 0.25,  # Responsiveness and upkeep
            "documentation": 0.20,  # How well documented (auto-assessed)
            "community": 0.15,  # Community size and health indicators
        }
        # Ensure weights sum to 1.0
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            logger.warning(f"Component weights do not sum to 1.0 (sum={sum(self.weights.values())}). Adjust weights.")

    def rate_repository(self, owner: str, repo: str) -> Optional[ResultData]:
        """Rate a GitHub repository and return the results."""
        logger.info(f"Analyzing repository: {owner}/{repo}")
        start_time = time.time()

        # Get basic repository info
        repo_data = self.data_collector.get_repo_info(owner, repo)
        if not repo_data:
            logger.error(f"Could not retrieve basic data for {owner}/{repo}. Aborting.")
            return None

        # --- Calculate Component Scores ---
        # Stars
        stars_comp = self.component_calculator.calculate_stars(repo_data)
        stars_comp.weight = self.weights["stars"]
        logger.info(f"Stars analysis complete: {stars_comp.score:.2f}")

        # Activity
        activity_comp = self.component_calculator.calculate_activity(owner, repo)
        activity_comp.weight = self.weights["activity"]
        logger.info(f"Activity analysis complete: {activity_comp.score:.2f}")

        # Maintenance
        maintenance_comp = self.component_calculator.calculate_maintenance(owner, repo, repo_data)
        maintenance_comp.weight = self.weights["maintenance"]
        logger.info(f"Maintenance analysis complete: {maintenance_comp.score:.2f}")

        # Documentation (now automated)
        documentation_comp = self.component_calculator.calculate_documentation(owner, repo, repo_data)
        documentation_comp.weight = self.weights["documentation"]
        logger.info(f"Documentation analysis complete: {documentation_comp.score:.2f}")

        # Community (now automated)
        community_comp = self.component_calculator.calculate_community(owner, repo, repo_data)
        community_comp.weight = self.weights["community"]
        logger.info(f"Community analysis complete: {community_comp.score:.2f}")
        # --- End Component Calculation ---

        # Calculate final score by weighted sum
        final_score = sum(
            comp.score * comp.weight
            for comp in [stars_comp, activity_comp, maintenance_comp, documentation_comp, community_comp]
        )

        # Ensure score is within bounds [0, 1]
        final_score = max(0, min(final_score, 1))

        rating = self._interpret_score(final_score)

        # Prepare results
        results = ResultData(
            repo=f"{owner}/{repo}",
            final_score=final_score,
            rating=rating,
            components={
                "stars": stars_comp,
                "activity": activity_comp,
                "maintenance": maintenance_comp,
                "documentation": documentation_comp,
                "community": community_comp,
            },
        )

        end_time = time.time()
        logger.info(f"Analysis for {owner}/{repo} took {end_time - start_time:.2f} seconds.")
        logger.info(f"Final Score: {final_score:.4f} - {rating}")
        return results

    @staticmethod
    def _interpret_score(score: float) -> str:
        """Interpret the final score into a qualitative rating."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.65:
            return "Very Good"
        elif score >= 0.5:
            return "Good"
        elif score >= 0.35:
            return "Fair"
        else:
            return "Poor"


class ResultsFormatter:
    """Formats and displays repository rating results."""

    @staticmethod
    def format_text(results: Optional[ResultData]) -> str:
        """Format results as multi-line text."""
        if not results:
            return "Unable to analyze repository."

        output: List[str] = []
        pad = 25  # Padding for alignment

        output.append("=" * 60)
        output.append(f" GITHUB REPOSITORY QUALITY SCORE: {results.repo}")
        output.append("=" * 60)
        output.append(f"{'Final Score:'.ljust(pad)} {results.final_score:.4f}")
        output.append(f"{'Rating:'.ljust(pad)} {results.rating}")
        output.append("-" * 60)
        output.append("Component Scores & Details:")
        output.append("-" * 60)

        for name, comp_data in results.components.items():
            output.append(
                f"[{name.capitalize()}]".ljust(pad)
                + f"Score: {comp_data.score:.3f} (Weight: {comp_data.weight * 100:.0f}%)"
            )

            data = comp_data.data
            if isinstance(data, dict) and "count" in data:  # Stars
                output.append(f"{'  Stars Count:'.ljust(pad)} {data['count']:,}")
            elif isinstance(data, ActivityData):
                output.append(
                    f"{'  Commit Freq Score:'.ljust(pad)} {data.commit_frequency_score:.3f} ({data.num_commits_6months} commits/6mo)"
                )
                output.append(
                    f"{'  Recent Activity Score:'.ljust(pad)} {data.recent_activity_score:.3f} ({data.days_since_last_commit} days ago)"
                )
                output.append(
                    f"{'  Release Freq Score:'.ljust(pad)} {data.release_frequency_score:.3f} ({data.num_releases_6months} releases/6mo)"
                )
            elif isinstance(data, MaintenanceData):
                output.append(
                    f"{'  Avg Issue Open Score:'.ljust(pad)} {data.avg_issue_open_time_score:.3f} ({data.num_open_issues} open)"
                )
                output.append(
                    f"{'  PR Merge Rate Score:'.ljust(pad)} {data.pr_merge_rate_score:.3f} (from {data.num_closed_prs_recent} recent closed PRs)"
                )
                output.append(
                    f"{'  Update Cadence Score:'.ljust(pad)} {data.update_cadence_score:.3f} ({data.days_since_last_update} days ago)"
                )
            elif isinstance(data, DocumentationData):
                output.append(
                    f"{'  README Score:'.ljust(pad)} {data.readme_score:.3f} (Exists: {data.has_readme}, Size: {data.readme_size})"
                )
                output.append(
                    f"{'  Guidelines Score:'.ljust(pad)} {data.contribution_guidelines_score:.3f} (CONTRIB: {data.has_contributing}, CoC: {data.has_code_of_conduct})"
                )
                output.append(
                    f"{'  Examples Score:'.ljust(pad)} {data.examples_score:.3f} (Exists: {data.has_examples})"
                )
                output.append(f"{'  Wiki Score:'.ljust(pad)} {data.wiki_score:.3f} (Enabled: {data.has_wiki})")
            elif isinstance(data, CommunityData):
                output.append(
                    f"{'  Contributor Growth Score:'.ljust(pad)} {data.contributor_growth_score:.3f} ({data.contributor_count} contributors)"
                )
                output.append(
                    f"{'  Issue Activity Score:'.ljust(pad)} {data.issue_activity_score:.3f} ({data.open_issues} open / {data.total_issues_approx} total est.)"
                )
                output.append(
                    f"{'  Community Health Score:'.ljust(pad)} {data.community_health_score:.3f} (Based on CONTRIB/CoC files)"
                )
            output.append("")  # Add spacing between components

        output.append("=" * 60)

        return "\n".join(output)

    @staticmethod
    def save_json(results: ResultData, filename: str) -> None:
        """Save results as JSON."""
        try:
            with open(filename, "w") as f:
                # Custom encoder to handle dataclasses and datetime if needed
                class EnhancedJSONEncoder(json.JSONEncoder):
                    def default(self, o):
                        if dataclasses.is_dataclass(o):
                            return dataclasses.asdict(o)
                        # Add handling for other types like datetime if they appear
                        # if isinstance(o, datetime.datetime):
                        #     return o.isoformat()
                        return super().default(o)

                json.dump(results, f, indent=2, cls=EnhancedJSONEncoder)
            logger.info(f"Results saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save results to {filename}: {e}")

    @staticmethod
    def print_results(results: Optional[ResultData]) -> None:
        """Print formatted results to console."""
        print(ResultsFormatter.format_text(results))


# --- Argument Parser and Main ---
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GitHub Repository Quality Score Calculator (Automated)")
    parser.add_argument("owner", help='Repository owner (e.g., "microsoft")')
    parser.add_argument("repo", help='Repository name (e.g., "vscode")')
    parser.add_argument("--output", help="Output file path for JSON results (e.g., results.json)")
    parser.add_argument("--quiet", action="store_true", help="Minimize console output (only show final score/rating)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    # Add flag to allow token prompt if needed, otherwise fully non-interactive
    # parser.add_argument('--prompt-token', action='store_true', help='Prompt for token if not found in args/env')

    return parser.parse_args()


@django_tasks.task()
def rate_repository(repo_id: int, owner: str, repo: str, *args, **kwargs) -> None:
    """Main function to run the GitHub repository rating tool."""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True
    )

    logger.info("--- GitHub Repository Quality Score (GRQS) Calculator ---")

    token = os.environ.get("GITHUB_TOKEN")

    # Initialize rater
    rater = RepoRater(token)

    # Rate repository
    results = rater.rate_repository(owner, repo)

    Rating.objects.create(repo_id=repo_id, rating=results.final_score)

    # Handle results
    formatter = ResultsFormatter()

    if results:
        formatter.print_results(results)
    else:
        logger.error(f"Failed to generate report for {owner}/{repo}.")
