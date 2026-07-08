from app.agents.ckan_registration.legacy_worker import secret_env_from_headers


def test_secret_headers_include_tapis_username_password_pass_through() -> None:
    env = secret_env_from_headers(
        {
            "request_headers": {
                "x-ckan-username": "alice",
                "x-ckan-password": "secret",
                "x-openai-api-key": "sk-test",
            }
        }
    )

    assert env["CKAN_AUTH_MODE"] == "tapis_password"
    assert env["CKAN_USERNAME"] == "alice"
    assert env["CKAN_PASSWORD"] == "secret"
    assert env["OPENAI_API_KEY"] == "sk-test"
