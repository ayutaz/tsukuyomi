"""Tests for authentication and rate limiting."""

import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import jwt
import pytest
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from src.server.auth import (
    AuthConfig,
    AuthManager,
    RateLimitMiddleware,
    User,
    authenticate,
    require_admin,
)


class TestAuthConfig:
    """Test AuthConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = AuthConfig()

        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiration_hours == 24
        assert config.require_api_key is True
        assert config.rate_limit_enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = AuthConfig(
            jwt_expiration_hours=48,
            rate_limit_max_requests=100,
            admin_api_key="test_admin_key",
        )

        assert config.jwt_expiration_hours == 48
        assert config.rate_limit_max_requests == 100
        assert config.admin_api_key == "test_admin_key"


class TestUser:
    """Test User class."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            api_key="test_key_123",
        )

        assert user.id == "user123"
        assert user.username == "testuser"
        assert user.is_active is True
        assert user.is_admin is False

    def test_user_to_dict(self):
        """Test user serialization."""
        user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            api_key="test_key",
            usage_quota=5000,
        )

        user_dict = user.to_dict()
        assert user_dict["id"] == "user123"
        assert user_dict["usage_quota"] == 5000
        assert "api_key" in user_dict


class TestAuthManager:
    """Test AuthManager class."""

    @pytest.fixture()
    def temp_db_path(self):
        """Create temporary database path."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_auth.db"
        yield str(db_path)
        import shutil

        shutil.rmtree(temp_dir)

    @pytest.fixture()
    def auth_config(self, temp_db_path):
        """Create test auth config."""
        return AuthConfig(
            auth_db_path=temp_db_path,
            rate_limit_enabled=False,  # Disable Redis for tests
            admin_api_key="admin_test_key",
        )

    @pytest.fixture()
    def auth_manager(self, auth_config):
        """Create auth manager."""
        return AuthManager(auth_config)

    def test_init_database(self, auth_manager, temp_db_path):
        """Test database initialization."""
        # Check database exists
        assert Path(temp_db_path).exists()

        # Check tables exist
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "users" in tables
        assert "usage_logs" in tables

        conn.close()

    def test_create_user(self, auth_manager):
        """Test creating a user."""
        user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            api_key="",  # Will be generated
        )

        created_user = auth_manager.create_user(user)

        assert created_user.id == "test_user"
        assert created_user.api_key.startswith("tsk_")
        assert len(created_user.api_key) > 20

    def test_duplicate_user(self, auth_manager):
        """Test creating duplicate user."""
        user = User(
            id="dup_user",
            username="dupuser",
            email="dup@example.com",
            api_key="dup_key",
        )

        auth_manager.create_user(user)

        # Try to create again
        with pytest.raises(ValueError, match="User already exists"):
            auth_manager.create_user(user)

    def test_get_user_by_api_key(self, auth_manager):
        """Test retrieving user by API key."""
        # Create user
        user = User(
            id="key_user",
            username="keyuser",
            email="key@example.com",
            api_key="test_api_key_123",
        )
        auth_manager.create_user(user)

        # Retrieve
        retrieved = auth_manager.get_user_by_api_key("test_api_key_123")

        assert retrieved is not None
        assert retrieved.id == "key_user"
        assert retrieved.username == "keyuser"

    def test_get_nonexistent_user(self, auth_manager):
        """Test retrieving non-existent user."""
        user = auth_manager.get_user_by_api_key("nonexistent_key")
        assert user is None

    def test_authenticate_with_api_key(self, auth_manager):
        """Test authentication with API key."""
        # Create user
        user = User(
            id="auth_user",
            username="authuser",
            email="auth@example.com",
            api_key="valid_api_key",
        )
        auth_manager.create_user(user)

        # Mock request
        request = Mock(spec=Request)

        # Authenticate
        authenticated = auth_manager.authenticate_request(
            request, api_key="valid_api_key"
        )

        assert authenticated.id == "auth_user"

    def test_authenticate_with_jwt(self, auth_manager):
        """Test authentication with JWT token."""
        # Create user
        user = User(
            id="jwt_user",
            username="jwtuser",
            email="jwt@example.com",
            api_key="jwt_key",
        )
        auth_manager.create_user(user)

        # Generate token
        token = auth_manager.generate_jwt_token(user)

        # Mock request and credentials
        request = Mock(spec=Request)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        # Authenticate
        authenticated = auth_manager.authenticate_request(request, token=credentials)

        assert authenticated.id == "jwt_user"

    def test_expired_jwt(self, auth_manager):
        """Test expired JWT token."""
        # Create expired token
        payload = {"user_id": "test", "exp": datetime.utcnow() - timedelta(hours=1)}
        expired_token = jwt.encode(
            payload,
            auth_manager.config.jwt_secret_key,
            algorithm=auth_manager.config.jwt_algorithm,
        )

        request = Mock(spec=Request)
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=expired_token
        )

        with pytest.raises(HTTPException) as exc_info:
            auth_manager.authenticate_request(request, token=credentials)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail

    def test_invalid_authentication(self, auth_manager):
        """Test invalid authentication."""
        request = Mock(spec=Request)

        with pytest.raises(HTTPException) as exc_info:
            auth_manager.authenticate_request(request, api_key="invalid_key")

        assert exc_info.value.status_code == 401

    def test_rate_limiting_memory(self, auth_manager):
        """Test in-memory rate limiting."""
        auth_manager.config.rate_limit_enabled = True
        auth_manager.config.rate_limit_max_requests = 5
        auth_manager.config.rate_limit_window_seconds = 1

        # Make requests
        for i in range(5):
            assert auth_manager.check_rate_limit("test_user", "/api/test") is True

        # 6th request should fail
        assert auth_manager.check_rate_limit("test_user", "/api/test") is False

        # Wait for window to pass
        time.sleep(1.1)

        # Should work again
        assert auth_manager.check_rate_limit("test_user", "/api/test") is True

    def test_rate_limiting_redis(self, auth_config):
        """Test Redis rate limiting."""
        auth_config.rate_limit_enabled = True

        with patch("redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True
            mock_client.zcard.return_value = 3

            auth_manager = AuthManager(auth_config)

            # Should allow request
            assert auth_manager.check_rate_limit("user", "/api") is True

            # Mock hitting limit
            mock_client.zcard.return_value = 100
            assert auth_manager.check_rate_limit("user", "/api") is False

    def test_log_usage(self, auth_manager, temp_db_path):
        """Test usage logging."""
        # Create user
        user = User(
            id="log_user",
            username="loguser",
            email="log@example.com",
            api_key="log_key",
        )
        auth_manager.create_user(user)

        # Log usage
        auth_manager.log_usage("log_user", "/api/synthesize", 125.5, 200)

        # Check log was created
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM usage_logs WHERE user_id = ?", ("log_user",))
        log = cursor.fetchone()

        assert log is not None
        assert log[1] == "log_user"  # user_id
        assert log[2] == "/api/synthesize"  # endpoint
        assert log[4] == 125.5  # response_time_ms

        # Check usage count was incremented
        cursor.execute("SELECT usage_count FROM users WHERE id = ?", ("log_user",))
        count = cursor.fetchone()[0]
        assert count == 1

        conn.close()

    def test_check_usage_quota(self, auth_manager):
        """Test usage quota checking."""
        user1 = User(
            id="quota_user1",
            username="quotauser1",
            email="quota1@example.com",
            api_key="quota_key1",
            usage_quota=100,
            usage_count=50,
        )

        user2 = User(
            id="quota_user2",
            username="quotauser2",
            email="quota2@example.com",
            api_key="quota_key2",
            usage_quota=100,
            usage_count=100,
        )

        assert auth_manager.check_usage_quota(user1) is True
        assert auth_manager.check_usage_quota(user2) is False

    def test_get_usage_stats(self, auth_manager, temp_db_path):
        """Test getting usage statistics."""
        # Create user and log some usage
        user = User(
            id="stats_user",
            username="statsuser",
            email="stats@example.com",
            api_key="stats_key",
        )
        auth_manager.create_user(user)

        # Log multiple requests
        endpoints = ["/api/synthesize", "/api/synthesize", "/api/morph", "/api/health"]
        for endpoint in endpoints:
            auth_manager.log_usage("stats_user", endpoint, 100.0, 200)

        # Get stats
        stats = auth_manager.get_usage_stats("stats_user", days=1)

        assert stats["total_requests"] > 0
        assert len(stats["endpoint_usage"]) > 0

        # Check endpoint counts
        endpoint_counts = {e["endpoint"]: e["count"] for e in stats["endpoint_usage"]}
        assert endpoint_counts["/api/synthesize"] == 2
        assert endpoint_counts["/api/morph"] == 1


class TestFastAPIDependencies:
    """Test FastAPI dependency functions."""

    @pytest.mark.asyncio()
    async def test_authenticate_dependency(self):
        """Test authenticate dependency."""
        with patch("src.server.auth.AuthManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            mock_user = User(
                id="dep_user",
                username="depuser",
                email="dep@example.com",
                api_key="dep_key",
            )
            mock_manager.authenticate_request.return_value = mock_user

            # Mock request
            request = Mock(spec=Request)

            # Test dependency
            user = await authenticate(
                request=request,
                api_key="test_key",
                token=None,
                auth_manager=mock_manager,
            )

            assert user == mock_user
            mock_manager.authenticate_request.assert_called_once()

    @pytest.mark.asyncio()
    async def test_require_admin_dependency(self):
        """Test require_admin dependency."""
        # Admin user
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@example.com",
            api_key="admin_key",
            is_admin=True,
        )

        # Should pass
        result = await require_admin(admin_user)
        assert result == admin_user

        # Non-admin user
        regular_user = User(
            id="regular",
            username="regular",
            email="regular@example.com",
            api_key="regular_key",
            is_admin=False,
        )

        # Should raise
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(regular_user)

        assert exc_info.value.status_code == 403
        assert "Admin access required" in exc_info.value.detail


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware class."""

    @pytest.mark.asyncio()
    async def test_rate_limit_middleware(self):
        """Test rate limiting middleware."""
        # Mock auth manager
        mock_auth_manager = MagicMock()
        mock_auth_manager.check_rate_limit.return_value = True
        mock_auth_manager.get_user_by_api_key.return_value = User(
            id="middleware_user",
            username="middleware",
            email="middleware@example.com",
            api_key="middleware_key",
        )

        middleware = RateLimitMiddleware(mock_auth_manager)

        # Mock request
        request = Mock(spec=Request)
        request.headers = {"X-API-Key": "test_key"}
        request.client.host = "127.0.0.1"
        request.url.path = "/api/test"

        # Mock call_next
        async def mock_call_next(req):
            response = Mock()
            response.status_code = 200
            return response

        # Test middleware
        response = await middleware(request, mock_call_next)

        assert response.status_code == 200
        mock_auth_manager.check_rate_limit.assert_called()

    @pytest.mark.asyncio()
    async def test_rate_limit_exceeded(self):
        """Test rate limit exceeded."""
        # Mock auth manager
        mock_auth_manager = MagicMock()
        mock_auth_manager.check_rate_limit.return_value = False

        middleware = RateLimitMiddleware(mock_auth_manager)

        # Mock request
        request = Mock(spec=Request)
        request.headers = {}
        request.client.host = "127.0.0.1"
        request.url.path = "/api/test"

        # Test middleware
        result = await middleware(request, None)

        assert isinstance(result, HTTPException)
        assert result.status_code == 429
        assert "Rate limit exceeded" in result.detail
