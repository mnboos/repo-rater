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

from .models import Rating

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
    examples_score: float  # Check for examples/samples/tests dir
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
                    reset_time = int(
                        response.headers.get("X-RateLimit-Reset", time.time() + 60)
                    )  # Default wait 60s if header missing
                    sleep_time = max(0, reset_time - time.time()) + 1  # Add buffer
                    logger.warning(f"Rate limit hit. Waiting for {sleep_time:.0f} seconds...")
                    time.sleep(sleep_time)
                    return self.make_request(url, params)  # Retry

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Specifically handle 404 Not Found without logging error unless debug
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response is not None
                and e.response.status_code == 404
            ):
                logger.debug(f"Resource not found (404): {url}")
                return None
            # Handle 451 Unavailable For Legal Reasons gracefully
            elif (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response is not None
                and e.response.status_code == 451
            ):
                logger.warning(f"Resource unavailable for legal reasons (451): {url}")
                return None
            # Log other errors
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

            # Handle potential None return from make_request after error/404/451
            if page_results is None:
                logger.warning(f"Received no results or error for page {page} of {url}. Stopping pagination.")
                break
            # Handle cases where the API returns a dict instead of a list (e.g., error message like 'abuse detection')
            if not isinstance(page_results, list):
                logger.warning(
                    f"Expected list but got {type(page_results)} for page {page} of {url}. Stopping pagination. Content: {page_results}"
                )
                # If it's a dict and contains 'message', log it.
                if isinstance(page_results, dict) and "message" in page_results:
                    logger.warning(f"GitHub API Message: {page_results['message']}")
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
        """Check if a file or directory exists at the given path using GitHub Contents API."""
        # Note: This check uses the Contents API, which works for files and directories.
        # A successful response (not 404) means the path exists.
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        logger.debug(f"Checking existence of path: {url}")
        response_data = self.make_request(url)
        exists = response_data is not None
        logger.debug(f"Path '{path}' exists: {exists}")
        return exists

    def is_path_directory(self, owner: str, repo: str, path: str) -> bool:
        """Check if a path points to a directory."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        logger.debug(f"Checking if path is directory: {url}")
        response_data = self.make_request(url)
        # The Contents API returns a list for directories, a dict for files.
        is_dir = isinstance(response_data, list)
        logger.debug(f"Path '{path}' is directory: {is_dir}")
        return is_dir


# --- Data Collector ---
class DataCollector:
    """Collects required data from GitHub for repository rating."""

    # Standard locations recognised by GitHub for community health files
    # See: https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/creating-a-default-community-health-file
    _COMMUNITY_FILE_LOCATIONS = ["", ".github/", "docs/"]  # Root, .github, docs

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
        # GitHub API might redirect for README, but /readme endpoint is canonical
        url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        readme_data = self.api_client.make_request(url)
        # Sometimes the main repo endpoint has readme info if /readme fails (e.g. permissions)
        # but this is less reliable. Stick to the dedicated endpoint.
        return readme_data  # Returns metadata including size

    def check_community_file_exists(self, owner: str, repo: str, base_filename: str) -> bool:
        """Check if a standard community file exists in root, .github/, or docs/."""
        # Check variations like CONTRIBUTING, CONTRIBUTING.md, contributing.md
        # Let's stick to the exact filename for now as specified by GitHub docs generally
        # We could add variations if needed: base_filename_no_ext = base_filename.split('.')[0]
        possible_filenames = [base_filename]  # Add variations here if needed

        for loc in self._COMMUNITY_FILE_LOCATIONS:
            for fname in possible_filenames:
                full_path = os.path.join(loc, fname).replace("\\", "/")  # Ensure forward slashes
                # Skip check for empty path component (root dir check handled correctly by API client)
                # if not full_path: continue
                if self.api_client.check_path_exists(owner, repo, full_path):
                    logger.info(f"Found community file '{base_filename}' at path: '{full_path}'")
                    return True
        logger.debug(f"Community file '{base_filename}' not found in standard locations.")
        return False

    def check_common_directory_exists(self, owner: str, repo: str, dir_names: List[str]) -> bool:
        """Check if any of the specified directory names exist at the root."""
        # This checks only root level for simplicity now. Could be expanded.
        for dir_name in dir_names:
            # Use is_path_directory to be sure it's a dir, not a file with the same name
            if self.api_client.is_path_directory(owner, repo, dir_name):
                logger.info(f"Found directory: '{dir_name}'")
                return True
        logger.debug(f"None of the directories {dir_names} found at repository root.")
        return False


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
        commits_per_month = total_commits / months_period if months_period > 0 else 0
        return min(commits_per_month / 50.0, 1.0)

    def calculate_recent_activity_score(self, last_commit_date_str: Optional[str]) -> Tuple[float, Optional[int]]:
        """Calculate recent activity score based on days since last commit."""
        if not last_commit_date_str:
            return 0.0, None
        try:
            last_commit_date = parse(last_commit_date_str)
            # Ensure timezone awareness for comparison
            if last_commit_date.tzinfo is None:
                last_commit_date = last_commit_date.replace(tzinfo=datetime.timezone.utc)  # Assume UTC if no tz
            days_since = (self.now - last_commit_date).days
            # Clamp negative days (e.g., clock skew, future commit date?) to 0
            days_since = max(0, days_since)
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
        recent_releases = []
        for r in releases:
            try:
                published_at = parse(r.get("published_at", ""))
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=datetime.timezone.utc)  # Assume UTC
                if published_at > cutoff_date:
                    recent_releases.append(r)
            except Exception as e:
                logger.debug(f"Could not parse release date {r.get('published_at')}: {e}")
                continue

        num_recent_releases = len(recent_releases)
        # Normalize based on releases per month, aiming for 1.0 around 1 release/month
        releases_per_month = num_recent_releases / months_period if months_period > 0 else 0
        return min(releases_per_month / 1.0, 1.0), num_recent_releases

    def calculate_avg_issue_open_time_score(self, open_issues: List[Dict[str, Any]]) -> float:
        """Calculate score based on average time issues have been open."""
        if not open_issues:
            return 1.0  # No open issues is good

        total_open_days = 0
        valid_issues = 0
        for issue in open_issues:
            if "pull_request" in issue and issue.get("pull_request"):  # Ensure it's not None
                continue  # Skip PRs listed as issues
            try:
                created_at = parse(issue["created_at"])
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                open_days = (self.now - created_at).days
                # Clamp negative days
                open_days = max(0, open_days)
                total_open_days += open_days
                valid_issues += 1
            except Exception as e:
                logger.debug(f"Could not parse issue created_at date {issue.get('created_at')}: {e}")
                continue

        if valid_issues == 0:
            return 0.75  # No valid non-PR issues to calculate from, neutral score

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
        merge_rate = merged_prs / total_closed if total_closed > 0 else 0
        # Directly use the merge rate as the score
        return max(0.0, min(merge_rate, 1.0)), total_closed

    def calculate_update_cadence_score(self, repo_updated_at_str: str) -> Tuple[float, int]:
        """Calculate score based on how recently the repo was updated."""
        try:
            repo_updated_at = parse(repo_updated_at_str)
            if repo_updated_at.tzinfo is None:
                repo_updated_at = repo_updated_at.replace(tzinfo=datetime.timezone.utc)
            days_since_update = (self.now - repo_updated_at).days
            days_since_update = max(0, days_since_update)
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
        if readme_size is not None and readme_size > 0:  # Ignore size 0
            # Add bonus for size (log scale), max bonus 0.5 for ~10KB+
            size_bonus = min(math.log10(readme_size) / 4.0, 0.5)  # log10(10000) = 4
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
        """Score based on presence of an examples/samples/tests directory."""
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
            # If only open issues exist, give a slightly lower score than perfect
            return 0.75 if open_issues_count > 0 else 1.0  # Perfect if 0 open, 0 total

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
        # Cache results of file checks within a single run
        self._check_cache: Dict[str, bool] = {}

    def _check_file_cached(self, owner: str, repo: str, filename: str) -> bool:
        """Helper to check community file existence with caching."""
        cache_key = f"file_{filename}"
        if cache_key not in self._check_cache:
            self._check_cache[cache_key] = self.data_collector.check_community_file_exists(owner, repo, filename)
        return self._check_cache[cache_key]

    def _check_dir_cached(self, owner: str, repo: str, dir_options: List[str]) -> bool:
        """Helper to check common directory existence with caching."""
        cache_key = f"dir_{'_'.join(sorted(dir_options))}"
        if cache_key not in self._check_cache:
            self._check_cache[cache_key] = self.data_collector.check_common_directory_exists(owner, repo, dir_options)
        return self._check_cache[cache_key]

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

        # Recent activity based on latest commit in the fetched list OR repo updated_at as fallback
        latest_commit_date_str = None
        if commits and len(commits) > 0:
            # Assume commits are sorted newest first by API default
            latest_commit_date_str = commits[0].get("commit", {}).get("committer", {}).get("date")

        # If no commits found in period, try fetching the very last commit regardless of date?
        # This might be redundant with repo update cadence. Let's rely on the period fetch for now.

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
        # Limit to avoid too much data, sort by updated desc is default
        open_issues_raw = self.data_collector.get_issues(owner, repo, state="open", limit=100)
        open_issues = [i for i in (open_issues_raw or []) if not i.get("pull_request")]  # Filter out PRs
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

        # Check for contribution files using enhanced check (checks root, .github/, docs/)
        # Use caching helper
        has_contributing = self._check_file_cached(owner, repo, "CONTRIBUTING.md")
        has_code_of_conduct = self._check_file_cached(owner, repo, "CODE_OF_CONDUCT.md")
        contribution_guidelines_score = self.metrics.calculate_contribution_guidelines_score(
            has_contributing, has_code_of_conduct
        )

        # Check for common examples/samples/tests directories at root using enhanced check
        # Django has 'tests', which often contain examples. Let's include 'tests'.
        # Use caching helper
        common_example_dirs = [
            "examples",
            "samples",
            "tests",
            "docs",
        ]  # Added 'tests' and 'docs' as potential indicators
        has_examples = self._check_dir_cached(owner, repo, common_example_dirs)
        examples_score = self.metrics.calculate_examples_score(has_examples)

        # Check for wiki
        has_wiki = repo_data.get("has_wiki", False)
        wiki_score = self.metrics.calculate_wiki_score(has_wiki)

        # Calculate overall documentation score (adjust weights as needed)
        documentation_score = (
            0.4 * readme_score + 0.3 * contribution_guidelines_score + 0.15 * examples_score + 0.15 * wiki_score
        )

        logger.debug(
            f"Documentation Checks: README={has_readme}({readme_size}), CONTRIB={has_contributing}, CoC={has_code_of_conduct}, Examples/Tests/Docs={has_examples}, Wiki={has_wiki}"
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
        # Filter out PRs counted as issues by GitHub API in this count if possible
        # The repo_data['open_issues_count'] includes PRs. We need a separate count of *just* open issues.
        # Fetch open issues again, but only need count. Limit 1 might work if > 0? No, need full count.
        # Reuse open_issues from maintenance calculation if possible? Risky if state differs.
        # Let's re-fetch open issues filtered:
        open_issues_raw = self.data_collector.get_issues(owner, repo, state="open", limit=500)  # Fetch more to be sure
        actual_open_issues_count = sum(1 for i in (open_issues_raw or []) if not i.get("pull_request"))
        logger.debug(
            f"Repo reports {open_issues_count} open issues/PRs. Found {actual_open_issues_count} actual open issues."
        )

        # Estimate total issues: Use actual_open_issues + closed issues count.
        closed_issues_raw = self.data_collector.get_issues(
            owner,
            repo,
            state="closed",
            limit=500,  # Fetch more recent closed issues
        )
        closed_issues_count = sum(1 for i in (closed_issues_raw or []) if not i.get("pull_request"))
        total_issues_approx = actual_open_issues_count + closed_issues_count

        # Fallback if no closed issues found but open issues exist
        if total_issues_approx == actual_open_issues_count and actual_open_issues_count > 0:
            # If we found > 100 closed items (incl PRs) perhaps it's just very active?
            # Let's make a more conservative guess if closed_issues_raw was full
            if len(closed_issues_raw or []) >= 500:
                total_issues_approx = actual_open_issues_count * 3  # Guess more closed if limit hit
            else:
                total_issues_approx = actual_open_issues_count * 2  # Simple guess otherwise

        issue_activity_score = self.metrics.calculate_issue_activity_score(
            actual_open_issues_count, total_issues_approx
        )

        # Community health score (based on guidelines files)
        # Reuse cached checks from documentation calculation
        has_contributing = self._check_file_cached(owner, repo, "CONTRIBUTING.md")
        has_code_of_conduct = self._check_file_cached(owner, repo, "CODE_OF_CONDUCT.md")
        community_health_score = self.metrics.calculate_community_health_score(has_contributing, has_code_of_conduct)

        # Calculate overall community score (adjust weights as needed)
        community_score = 0.4 * contributor_growth_score + 0.3 * issue_activity_score + 0.3 * community_health_score

        logger.debug(
            f"Community: Contribs={contributor_count}, ActualOpenIssues={actual_open_issues_count}, TotalIssuesEst={total_issues_approx}, ClosedIssuesFound={closed_issues_count}"
        )
        logger.debug(
            f"Community Scores: Growth={contributor_growth_score:.2f}, IssueActivity={issue_activity_score:.2f}, HealthFiles={community_health_score:.2f} -> Overall: {community_score:.2f}"
        )

        community_data = CommunityData(
            contributor_count=contributor_count,
            contributor_growth_score=contributor_growth_score,
            issue_activity_score=issue_activity_score,
            community_health_score=community_health_score,
            open_issues=actual_open_issues_count,  # Use actual count
            total_issues_approx=total_issues_approx,
        )
        # Reset cache after finishing analysis for one repo
        self._check_cache = {}
        return ComponentData(score=community_score, weight=0, data=community_data)


# --- Input Provider (Simplified) ---
# InputProvider class remains the same as it only handles token/repo args


class InputProvider:
    """Handles user input for non-metric configuration like token and repo."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize with command line arguments."""
        self.args = args

    def get_github_token(self) -> Optional[str]:
        """Get GitHub token from args, environment, or prompt."""
        # 1. Command line argument
        if hasattr(self.args, "token") and self.args.token:
            logger.debug("Using token from command line argument.")
            return self.args.token

        # 2. Environment variable
        env_token = os.environ.get("GITHUB_TOKEN")
        if env_token:
            logger.debug("Found token in GITHUB_TOKEN environment variable.")
            return env_token  # Use directly for automation

        # 3. Prompt user (optional, only if explicitly enabled)
        use_token_prompt = getattr(self.args, "prompt_token", False)
        if use_token_prompt:
            logger.info("No token found in args or env. Prompting for token.")
            try:
                # Check if running in interactive terminal before prompting
                if os.isatty(0):
                    use_token = input("Use a GitHub token? (y/n) [n]: ").lower() == "y"
                    if use_token:
                        return getpass("Enter GitHub token: ")
                else:
                    logger.warning("Non-interactive terminal detected, cannot prompt for token.")
            except EOFError:
                logger.warning("EOFError encountered, likely running non-interactively. Cannot prompt for token.")
        else:
            logger.warning("No GitHub token provided via args or environment. API rate limits will be stricter.")

        return None


# --- Repo Rater ---
class RepoRater:
    """Main class for rating GitHub repositories."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the repository rater with dependencies."""
        self.api_client = GitHubAPIClient(token)
        self.data_collector = DataCollector(self.api_client)
        self.metrics = MetricsCalculator()
        # Pass the data collector and metrics instances to component calculator
        self.component_calculator = ComponentCalculator(self.data_collector, self.metrics)

        # Component weights (Adjust these based on the perceived importance of each automated component)
        self.weights: Dict[str, float] = {
            "stars": 0.15,  # Popularity indicator
            "activity": 0.25,  # Development momentum
            "maintenance": 0.25,  # Responsiveness and upkeep
            "documentation": 0.20,  # How well documented (auto-assessed)
            "community": 0.15,  # Community size and health indicators
        }
        # Ensure weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Component weights do not sum to 1.0 (sum={weight_sum}). Adjust weights.")
            # Optional: Normalize weights if they don't sum to 1
            # norm_factor = 1.0 / weight_sum if weight_sum != 0 else 0
            # self.weights = {k: v * norm_factor for k, v in self.weights.items()}
            # logger.warning("Weights have been normalized.")

    def rate_repository(self, owner: str, repo: str) -> Optional[ResultData]:
        """Rate a GitHub repository and return the results."""
        logger.info(f"Analyzing repository: {owner}/{repo}")
        start_time = time.time()

        # Reset component calculator cache for new repo analysis
        self.component_calculator._check_cache = {}

        # Get basic repository info
        repo_data = self.data_collector.get_repo_info(owner, repo)
        if not repo_data:
            logger.error(f"Could not retrieve basic data for {owner}/{repo}. Aborting.")
            return None
        # Handle disabled repositories
        if repo_data.get("disabled"):
            logger.warning(f"Repository {owner}/{repo} is disabled. Analysis may be incomplete or inaccurate.")
            # Optionally return a specific result or score for disabled repos

        # --- Calculate Component Scores ---
        all_components: Dict[str, ComponentData] = {}
        try:
            # Stars
            stars_comp = self.component_calculator.calculate_stars(repo_data)
            stars_comp.weight = self.weights["stars"]
            all_components["stars"] = stars_comp
            logger.info(f"Stars analysis complete: {stars_comp.score:.2f}")

            # Activity
            activity_comp = self.component_calculator.calculate_activity(owner, repo)
            activity_comp.weight = self.weights["activity"]
            all_components["activity"] = activity_comp
            logger.info(f"Activity analysis complete: {activity_comp.score:.2f}")

            # Maintenance
            maintenance_comp = self.component_calculator.calculate_maintenance(owner, repo, repo_data)
            maintenance_comp.weight = self.weights["maintenance"]
            all_components["maintenance"] = maintenance_comp
            logger.info(f"Maintenance analysis complete: {maintenance_comp.score:.2f}")

            # Documentation (now automated with better checks)
            documentation_comp = self.component_calculator.calculate_documentation(owner, repo, repo_data)
            documentation_comp.weight = self.weights["documentation"]
            all_components["documentation"] = documentation_comp
            logger.info(f"Documentation analysis complete: {documentation_comp.score:.2f}")

            # Community (now automated with better checks)
            community_comp = self.component_calculator.calculate_community(owner, repo, repo_data)
            community_comp.weight = self.weights["community"]
            all_components["community"] = community_comp
            logger.info(f"Community analysis complete: {community_comp.score:.2f}")

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during component calculation for {owner}/{repo}: {e}", exc_info=True
            )
            # Decide how to handle: return partial results or None? Returning None for now.
            return None
        # --- End Component Calculation ---

        # Calculate final score by weighted sum
        final_score = sum(comp.score * comp.weight for comp in all_components.values())

        # Ensure score is within bounds [0, 1]
        final_score = max(0, min(final_score, 1))

        rating = self._interpret_score(final_score)

        # Prepare results
        results = ResultData(
            repo=f"{owner}/{repo}",
            final_score=final_score,
            rating=rating,
            components=all_components,
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


# --- Results Formatter ---
class ResultsFormatter:
    """Formats and displays repository rating results."""

    @staticmethod
    def format_text(results: Optional[ResultData]) -> str:
        """Format results as multi-line text."""
        if not results:
            return "Unable to analyze repository or analysis failed."

        output: List[str] = []
        pad = 28  # Adjusted padding for potentially longer labels

        output.append("=" * 60)
        output.append(f" GITHUB REPOSITORY QUALITY SCORE: {results.repo}")
        output.append("=" * 60)
        output.append(f"{'Final Score:'.ljust(pad)} {results.final_score:.4f}")
        output.append(f"{'Rating:'.ljust(pad)} {results.rating}")
        output.append("-" * 60)
        output.append("Component Scores & Details:")
        output.append("-" * 60)

        # Ensure consistent order if needed
        component_order = ["stars", "activity", "maintenance", "documentation", "community"]
        components_to_print = {k: results.components[k] for k in component_order if k in results.components}
        # Add any missing components (e.g., if weights change)
        components_to_print.update({k: v for k, v in results.components.items() if k not in components_to_print})

        for name, comp_data in components_to_print.items():
            output.append(
                f"[{name.capitalize()}]".ljust(pad)
                + f"Score: {comp_data.score:.3f} (Weight: {comp_data.weight * 100:.0f}%)"
            )

            data = comp_data.data
            # Use isinstance checks with the specific dataclasses
            if name == "stars" and isinstance(data, dict) and "count" in data:  # Stars special case
                output.append(f"{'  Stars Count:'.ljust(pad)} {data['count']:,}")
            elif isinstance(data, ActivityData):
                output.append(
                    f"{'  Commit Freq Score:'.ljust(pad)} {data.commit_frequency_score:.3f} ({data.num_commits_6months} commits/6mo)"
                )
                days_ago = (
                    f"{data.days_since_last_commit} days ago" if data.days_since_last_commit is not None else "N/A"
                )
                output.append(f"{'  Recent Activity Score:'.ljust(pad)} {data.recent_activity_score:.3f} ({days_ago})")
                output.append(
                    f"{'  Release Freq Score:'.ljust(pad)} {data.release_frequency_score:.3f} ({data.num_releases_6months} releases/6mo)"
                )
            elif isinstance(data, MaintenanceData):
                output.append(
                    f"{'  Avg Issue Open Score:'.ljust(pad)} {data.avg_issue_open_time_score:.3f} ({data.num_open_issues} open issues)"
                )
                output.append(
                    f"{'  PR Merge Rate Score:'.ljust(pad)} {data.pr_merge_rate_score:.3f} (from {data.num_closed_prs_recent} recent closed PRs)"
                )
                output.append(
                    f"{'  Update Cadence Score:'.ljust(pad)} {data.update_cadence_score:.3f} ({data.days_since_last_update} days ago)"
                )
            elif isinstance(data, DocumentationData):
                readme_size_str = f"{data.readme_size:,}" if data.readme_size is not None else "N/A"
                output.append(
                    f"{'  README Score:'.ljust(pad)} {data.readme_score:.3f} (Exists: {data.has_readme}, Size: {readme_size_str})"
                )
                output.append(
                    f"{'  Guidelines Score:'.ljust(pad)} {data.contribution_guidelines_score:.3f} (CONTRIB: {data.has_contributing}, CoC: {data.has_code_of_conduct})"
                )
                output.append(
                    f"{'  Examples/Tests Score:'.ljust(pad)} {data.examples_score:.3f} (Common Dir Exists: {data.has_examples})"  # Updated label
                )
                output.append(f"{'  Wiki Score:'.ljust(pad)} {data.wiki_score:.3f} (Enabled: {data.has_wiki})")
            elif isinstance(data, CommunityData):
                output.append(
                    f"{'  Contributor Growth Score:'.ljust(pad)} {data.contributor_growth_score:.3f} ({data.contributor_count} contributors)"
                )
                # Use the actual open issues count here
                output.append(
                    f"{'  Issue Activity Score:'.ljust(pad)} {data.issue_activity_score:.3f} ({data.open_issues} open / {data.total_issues_approx} total est.)"
                )
                output.append(
                    f"{'  Community Health Score:'.ljust(pad)} {data.community_health_score:.3f} (Based on CONTRIB/CoC files)"
                )
            else:
                # Fallback for unexpected data types
                output.append(f"{'  Data:'.ljust(pad)} {data!s}")

            output.append("")  # Add spacing between components

        output.append("=" * 60)

        return "\n".join(output)

    @staticmethod
    def save_json(results: ResultData, filename: str) -> None:
        """Save results as JSON."""
        try:
            with open(filename, "w", encoding="utf-8") as f:  # Specify encoding
                # Custom encoder to handle dataclasses
                class EnhancedJSONEncoder(json.JSONEncoder):
                    def default(self, o):
                        if dataclasses.is_dataclass(o):
                            return dataclasses.asdict(o)
                        # Add handling for other non-serializable types if they appear
                        # if isinstance(o, datetime.datetime):
                        #     return o.isoformat()
                        return super().default(o)

                json.dump(results, f, indent=2, cls=EnhancedJSONEncoder, ensure_ascii=False)
            logger.info(f"Results saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save results to {filename}: {e}")
        except TypeError as e:
            logger.error(f"Failed to serialize results to JSON: {e}")

    @staticmethod
    def print_results(results: Optional[ResultData]) -> None:
        """Print formatted results to console."""
        print(ResultsFormatter.format_text(results))


@django_tasks.task()
def rate_repository(repo_id: int, owner: str, repo: str, *args, **kwargs) -> None:
    """
    Django Task wrapper to rate a GitHub repository and save the result.
    Renamed from 'rate_repository' to avoid conflict with the class method.
    """

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True
    )
    task_logger = logging.getLogger("rate_repository_task")  # Use a specific logger name

    logger.info("--- GitHub Repository Quality Score (GRQS) Calculator ---")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        task_logger.warning("GITHUB_TOKEN environment variable not set. Rate limits will be stricter.")

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
