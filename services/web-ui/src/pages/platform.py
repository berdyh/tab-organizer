"""Platform account, company discovery, B2B, and maintainer page."""

from typing import Any, Optional

import requests
import streamlit as st

from ..api.client import SyncAPIClient


ACCOUNT_SCOPED_STATE_KEYS = (
    "platform_token",
    "platform_profile",
    "platform_dashboard",
    "platform_first_call",
    "platform_first_call_result",
    "platform_token_list",
    "platform_created_token",
    "platform_last_raw_api_token",
    "platform_first_call_api_token",
    "platform_company_results",
    "platform_selected_company",
    "platform_issues",
)


def _clear_account_scoped_state() -> None:
    for key in ACCOUNT_SCOPED_STATE_KEYS:
        st.session_state.pop(key, None)


def _extract_token(response: dict[str, Any]) -> Optional[str]:
    for key in ("access_token", "session_token", "token", "jwt", "id_token"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value

    auth = response.get("auth")
    if isinstance(auth, dict):
        return _extract_token(auth)

    return None


def _extract_profile(response: dict[str, Any]) -> dict[str, Any]:
    for key in ("user", "profile", "account"):
        value = response.get(key)
        if isinstance(value, dict):
            return value

    profile_keys = {
        "id",
        "email",
        "name",
        "role",
        "roles",
        "account_type",
        "accountType",
        "company_id",
        "companyId",
        "is_maintainer",
        "isMaintainer",
    }
    if isinstance(response, dict) and profile_keys.intersection(response):
        return response

    return {}


def _roles(profile: dict[str, Any]) -> set[str]:
    values = []
    for key in ("role", "roles", "permissions"):
        value = profile.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return {value.lower() for value in values}


def _account_type(profile: dict[str, Any]) -> str:
    value = profile.get("account_type") or profile.get("accountType") or ""
    return str(value).lower()


def _is_b2b_user(profile: dict[str, Any]) -> bool:
    roles = _roles(profile)
    account_type = _account_type(profile)
    return (
        account_type in {"business", "b2b", "team", "enterprise"}
        or bool(roles & {"business", "b2b", "team", "enterprise"})
        or bool(profile.get("company_id") or profile.get("companyId"))
    )


def _is_maintainer(profile: dict[str, Any]) -> bool:
    roles = _roles(profile)
    return bool(roles & {"admin", "maintainer"}) or bool(
        profile.get("is_maintainer") or profile.get("isMaintainer")
    )


def _items_from_response(response: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if not isinstance(response, dict):
        return []

    for key in keys:
        value = response.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _first_text(
    data: dict[str, Any], *keys: str, default: str = "Not available"
) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return str(value)
    return default


def _platform_error(prefix: str, error: Exception) -> None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    detail = _response_error_detail(response)
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
        st.warning(f"{prefix}: platform service is unavailable.")
        return
    if status_code in {404, 405}:
        message = "the backend endpoint is not available"
        if detail:
            message = f"{message} ({detail})"
        st.warning(f"{prefix}: {message}.")
        return
    if isinstance(error, requests.HTTPError) and status_code:
        message = f"backend returned HTTP {status_code}"
        if detail:
            message = f"{message}: {detail}"
        st.warning(f"{prefix}: {message}.")
        return
    st.warning(f"{prefix}: {error}")


def _response_error_detail(response: Any) -> str:
    """Extract a short user-facing detail from a backend error response."""
    if response is None:
        return ""

    payload = None
    try:
        payload = response.json()
    except (AttributeError, ValueError):
        payload = None

    detail = ""
    if isinstance(payload, dict):
        value = payload.get("detail") or payload.get("message") or payload.get("error")
        detail = str(value) if value else ""
    elif payload is not None:
        detail = str(payload)

    if not detail:
        text = getattr(response, "text", "")
        detail = str(text).strip()

    if len(detail) > 240:
        return f"{detail[:237]}..."
    return detail


def _store_auth_response(response: dict[str, Any]) -> bool:
    token = _extract_token(response)
    profile = _extract_profile(response)

    if token:
        _clear_account_scoped_state()
        st.session_state.platform_token = token
    if profile:
        st.session_state.platform_profile = profile

    return bool(token)


def _extract_raw_api_token(response: dict[str, Any]) -> str:
    for key in ("token", "apiKey", "api_key", "value"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _format_profile_label(profile: dict[str, Any]) -> str:
    return _first_text(profile, "email", "name", "id", default="signed-in user")


def render_platform_page() -> None:
    """Render the platform user and maintainer surface."""
    st.header("Platform")

    if "api_client" not in st.session_state:
        st.session_state.api_client = SyncAPIClient()

    api = st.session_state.api_client
    token = st.session_state.get("platform_token")
    profile = st.session_state.get("platform_profile", {})

    _render_auth_panel(api, token, profile)
    st.divider()

    token = st.session_state.get("platform_token")
    profile = st.session_state.get("platform_profile", {})
    _render_company_search(api, token)
    st.divider()

    _render_b2b_panel(api, token, profile)
    st.divider()

    _render_maintainer_panel(api, token, profile)


def _render_auth_panel(
    api: SyncAPIClient,
    token: Optional[str],
    profile: dict[str, Any],
) -> None:
    st.subheader("Account Access")

    if token:
        st.success(f"Signed in as {_format_profile_label(profile)}")
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Refresh Profile", key="platform_refresh_profile"):
                try:
                    st.session_state.platform_profile = _extract_profile(
                        api.platform_me(token)
                    )
                    st.success("Profile refreshed")
                except Exception as error:
                    _platform_error("Could not load profile", error)

        with col2:
            if st.button("Sign Out", key="platform_sign_out"):
                _clear_account_scoped_state()
                st.rerun()
        return

    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

    with login_tab:
        with st.form("platform_login_form"):
            email = st.text_input("Email", key="platform_login_email")
            password = st.text_input(
                "Password",
                type="password",
                key="platform_login_password",
            )
            submitted = st.form_submit_button("Log In", type="primary")

        if submitted:
            if not email or not password:
                st.warning("Email and password are required.")
            else:
                try:
                    response = api.platform_login(email, password)
                    if _store_auth_response(response):
                        st.success("Logged in")
                        st.rerun()
                    else:
                        st.warning("Login response did not include an access token.")
                except Exception as error:
                    _platform_error("Login failed", error)

    with signup_tab:
        with st.form("platform_signup_form"):
            name = st.text_input("Name", key="platform_signup_name")
            email = st.text_input("Email", key="platform_signup_email")
            password = st.text_input(
                "Password",
                type="password",
                key="platform_signup_password",
            )
            account_type = st.selectbox(
                "Account type",
                options=["individual", "business", "maintainer"],
                key="platform_signup_account_type",
            )
            company_name = st.text_input(
                "Company name",
                key="platform_signup_company_name",
            )
            maintainer_code = ""
            if account_type == "maintainer":
                maintainer_code = st.text_input(
                    "Maintainer code",
                    type="password",
                    key="platform_signup_maintainer_code",
                )
            submitted = st.form_submit_button("Create Account", type="primary")

        if submitted:
            if not name or not email or not password:
                st.warning("Name, email, and password are required.")
            elif account_type == "business" and not company_name:
                st.warning("Company name is required for business accounts.")
            elif account_type == "maintainer" and not maintainer_code:
                st.warning("Maintainer code is required for maintainer accounts.")
            else:
                try:
                    response = api.platform_signup(
                        email=email,
                        password=password,
                        name=name,
                        account_type=account_type,
                        company_name=company_name
                        if account_type == "business"
                        else None,
                        role="maintainer" if account_type == "maintainer" else None,
                        maintainer_code=maintainer_code
                        if account_type == "maintainer"
                        else None,
                    )
                    if _store_auth_response(response):
                        st.success("Account created")
                        st.rerun()
                    else:
                        st.success("Account created. Log in to continue.")
                except Exception as error:
                    _platform_error("Sign up failed", error)


def _render_company_search(api: SyncAPIClient, token: Optional[str]) -> None:
    st.subheader("Company Search")

    if not token:
        st.info("Log in to search companies.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search companies",
            placeholder="Company name, domain, or identifier",
            key="platform_company_query",
        )
    with col2:
        limit = st.number_input(
            "Results",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="platform_company_limit",
        )

    if st.button("Find Companies", key="platform_find_companies", type="primary"):
        if not query:
            st.warning("Enter a company search query.")
        else:
            try:
                response = api.search_companies(
                    query=query, limit=int(limit), token=token
                )
                st.session_state.platform_company_results = response
            except Exception as error:
                _platform_error("Company search failed", error)

    results = _items_from_response(
        st.session_state.get("platform_company_results"),
        "companies",
        "results",
        "items",
    )
    if results:
        st.write(f"Found {len(results)} companies")
        for index, company in enumerate(results):
            _render_company_row(api, token, company, index)
    elif st.session_state.get("platform_company_results") is not None:
        st.info("No companies matched that search.")

    selected = st.session_state.get("platform_selected_company")
    if isinstance(selected, dict) and selected:
        st.markdown("#### Company Detail")
        _render_key_value_summary(
            selected,
            ["id", "name", "domain", "website", "industry", "size", "status"],
        )


def _render_company_row(
    api: SyncAPIClient,
    token: Optional[str],
    company: dict[str, Any],
    index: int,
) -> None:
    company_id = _first_text(
        company, "id", "company_id", "companyId", "slug", default=""
    )
    name = _first_text(
        company, "name", "legal_name", "domain", default="Unnamed company"
    )
    subtitle = _first_text(company, "domain", "website", "industry", default="")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f"**{name}**")
        if subtitle:
            st.caption(subtitle)
    with col2:
        if st.button(
            "View",
            key=f"platform_company_view_{company_id or index}",
            disabled=not company_id,
        ):
            try:
                response = api.get_company(
                    company_id,
                    token=token,
                )
                st.session_state.platform_selected_company = response.get(
                    "company", response
                )
            except Exception as error:
                _platform_error("Company lookup failed", error)


def _render_b2b_panel(
    api: SyncAPIClient,
    token: Optional[str],
    profile: dict[str, Any],
) -> None:
    st.subheader("B2B API")

    if not token:
        st.info("Log in with a business account to manage API tokens.")
        return

    if profile and not _is_b2b_user(profile):
        st.info(
            "Business account access is required for API tokens and dashboard data."
        )
        return

    if not profile:
        st.info("Profile is not loaded. The backend will validate business access.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Refresh Dashboard", key="platform_refresh_dashboard"):
            try:
                st.session_state.platform_dashboard = api.get_platform_dashboard(token)
            except Exception as error:
                _platform_error("Dashboard refresh failed", error)
    with col2:
        if st.button("Refresh Tokens", key="platform_refresh_tokens"):
            try:
                st.session_state.platform_token_list = api.list_b2b_tokens(token)
            except Exception as error:
                _platform_error("Token list refresh failed", error)
    with col3:
        if st.button("Load First API Call", key="platform_load_first_call"):
            try:
                st.session_state.platform_first_call = api.get_b2b_first_call(token)
            except Exception as error:
                _platform_error("First API call guidance failed", error)

    dashboard = st.session_state.get("platform_dashboard")
    if isinstance(dashboard, dict) and dashboard:
        _render_dashboard(dashboard)

    first_call = st.session_state.get("platform_first_call")
    if isinstance(first_call, dict) and first_call:
        _render_first_call(first_call)

    _render_token_list(api, token)

    with st.form("platform_b2b_token_form"):
        token_name = st.text_input(
            "Token name",
            value="Production integration",
            key="platform_b2b_token_name",
        )
        scopes = st.multiselect(
            "Scopes",
            options=["companies:read"],
            default=["companies:read"],
            key="platform_b2b_token_scopes",
        )
        submitted = st.form_submit_button("Create API Token")

    if submitted:
        if not token_name:
            st.warning("Token name is required.")
        else:
            try:
                response = api.create_b2b_token(token_name, scopes=scopes, token=token)
                st.session_state.platform_created_token = response
                raw_api_token = _extract_raw_api_token(response)
                if raw_api_token:
                    st.session_state.platform_last_raw_api_token = raw_api_token
                    st.session_state.platform_first_call_api_token = raw_api_token
                st.session_state.platform_token_list = api.list_b2b_tokens(token)
                st.success("API token created")
            except Exception as error:
                _platform_error("Token creation failed", error)

    created = st.session_state.get("platform_created_token")
    if isinstance(created, dict) and created:
        api_token = _extract_raw_api_token(created)
        if api_token:
            st.code(api_token)
        else:
            _render_key_value_summary(created, ["id", "name", "status", "created_at"])

    _render_first_call_runner(api, token)


def _render_token_list(api: SyncAPIClient, token: str) -> None:
    token_list = st.session_state.get("platform_token_list")
    tokens = _items_from_response(token_list, "tokens", "api_tokens", "items")
    if not tokens:
        if token_list is not None:
            st.info("No API tokens found.")
        return

    st.markdown("#### API Tokens")
    for index, item in enumerate(tokens):
        token_id = _first_text(item, "id", "token_id", "tokenId", default="")
        name = _first_text(item, "name", default="Unnamed token")
        prefix = _first_text(item, "prefix", default="")
        status = _first_text(item, "status", default="unknown")
        scopes = item.get("scopes") if isinstance(item.get("scopes"), list) else []

        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{name}**")
            caption_parts = [status]
            if prefix:
                caption_parts.append(prefix)
            if scopes:
                caption_parts.append(", ".join(str(scope) for scope in scopes))
            st.caption(" | ".join(caption_parts))
        with col2:
            if st.button(
                "Revoke",
                key=f"platform_revoke_token_{token_id or index}",
                disabled=not token_id or status == "revoked",
            ):
                try:
                    api.revoke_b2b_token(token_id, token=token)
                    st.session_state.platform_token_list = api.list_b2b_tokens(token)
                    st.success("API token revoked")
                    st.rerun()
                except Exception as error:
                    _platform_error("Token revoke failed", error)


def _render_dashboard(dashboard: dict[str, Any]) -> None:
    metrics = _dashboard_metrics(dashboard)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("API Calls", metrics["api_calls"])
    with col2:
        st.metric("Active Tokens", metrics["active_tokens"])
    with col3:
        st.metric("Companies", metrics["companies"])
    with col4:
        st.metric("Issues", metrics["issues"])


def _dashboard_metrics(dashboard: dict[str, Any]) -> dict[str, Any]:
    metrics = dashboard.get("metrics")
    if not isinstance(metrics, dict):
        metrics = dashboard.get("counters")
    if not isinstance(metrics, dict):
        metrics = dashboard

    return {
        "api_calls": metrics.get(
            "api_requests_total", metrics.get("api_calls", metrics.get("calls", 0))
        ),
        "active_tokens": metrics.get(
            "api_tokens_active", metrics.get("active_tokens", 0)
        ),
        "companies": metrics.get(
            "companies_available",
            metrics.get("companies", metrics.get("company_count", 0)),
        ),
        "issues": metrics.get(
            "issues_created", metrics.get("issues", metrics.get("open_issues", 0))
        ),
    }


def _render_first_call(first_call: dict[str, Any]) -> None:
    st.markdown("#### First API Call")
    description = _first_text(first_call, "description", "message", default="")
    if description:
        st.caption(description)

    command = _first_text(first_call, "curl", "command", "example", default="")
    if command:
        st.code(command, language="bash")
    else:
        endpoint = _first_text(
            first_call, "endpoint", default="/platform/companies/search"
        )
        st.code(f"GET {endpoint}\nAuthorization: Bearer <api-token>", language="http")


def _render_first_call_runner(api: SyncAPIClient, session_token: str) -> None:
    st.markdown("#### Test API Call")
    if "platform_first_call_api_token" not in st.session_state:
        st.session_state.platform_first_call_api_token = st.session_state.get(
            "platform_last_raw_api_token", ""
        )

    api_token = st.text_input(
        "API token",
        type="password",
        key="platform_first_call_api_token",
    )
    query = st.text_input(
        "Company query",
        value="acme",
        key="platform_first_call_query",
    )

    if st.button("Run First API Call", key="platform_run_first_call"):
        if not api_token:
            st.warning("API token is required.")
        else:
            try:
                response = api.search_companies_with_api_token(
                    query=query or "acme",
                    api_token=api_token,
                    limit=5,
                )
                st.session_state.platform_first_call_result = response
                st.session_state.platform_dashboard = api.get_platform_dashboard(
                    session_token
                )
            except Exception as error:
                _platform_error("First API call failed", error)

    result = st.session_state.get("platform_first_call_result")
    if result is not None:
        companies = _items_from_response(result, "companies", "results", "items")
        if companies:
            st.write(f"First call returned {len(companies)} companies")
            for company in companies[:5]:
                st.write(
                    f"**{_first_text(company, 'name', default='Company')}**"
                )
                st.caption(_first_text(company, "domain", "industry", default=""))
        else:
            st.info("First API call returned no companies.")


def _render_maintainer_panel(
    api: SyncAPIClient,
    token: Optional[str],
    profile: dict[str, Any],
) -> None:
    st.subheader("Maintainer Issues")

    if not token:
        st.info("Log in with a maintainer account to view issues.")
        return

    if not _is_maintainer(profile):
        st.info("Maintainer access is required to view platform issues.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        status = st.selectbox(
            "Status",
            options=["open", "triaged", "closed", "all"],
            key="platform_issue_status",
        )
    with col2:
        if st.button("Load Issues", key="platform_load_issues"):
            try:
                selected_status = None if status == "all" else status
                st.session_state.platform_issues = api.get_maintainer_issues(
                    status=selected_status,
                    token=token,
                )
            except Exception as error:
                _platform_error("Issue load failed", error)

    issues = _items_from_response(
        st.session_state.get("platform_issues"),
        "issues",
        "results",
        "items",
    )
    if issues:
        st.write(f"{len(issues)} issues")
        for index, issue in enumerate(issues):
            _render_issue(issue, index)
    elif st.session_state.get("platform_issues") is not None:
        st.info("No maintainer issues matched the selected status.")


def _render_issue(issue: dict[str, Any], index: int) -> None:
    issue_id = _first_text(issue, "id", "issue_id", "issueId", default=str(index + 1))
    title = _first_text(issue, "title", "summary", "message", default="Untitled issue")
    status = _first_text(issue, "status", "state", default="unknown")

    with st.expander(f"{title} ({status})"):
        st.caption(f"ID: {issue_id}")
        _render_key_value_summary(
            issue,
            ["severity", "source", "company_id", "created_at", "updated_at"],
        )
        details = _first_text(issue, "details", "description", "body", default="")
        if details:
            st.write(details)


def _render_key_value_summary(data: dict[str, Any], keys: list[str]) -> None:
    rows = []
    for key in keys:
        if key in data and data[key] is not None:
            rows.append({"field": key, "value": str(data[key])})

    if rows:
        st.table(rows)
