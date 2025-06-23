"""
Authentication and Rate Limiting for Tsukuyomi TTS API

Features:
- API key authentication
- JWT token support
- Rate limiting per user/IP
- Usage tracking and quotas
- Admin management interface
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from fastapi import HTTPException, Security, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel
import redis
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass
import logging
import time
from functools import wraps
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class AuthConfig(BaseModel):
    """Authentication configuration."""
    # JWT settings
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # API key settings
    api_key_header_name: str = "X-API-Key"
    require_api_key: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_max_requests: int = 60
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Database settings
    auth_db_path: str = "data/auth.db"
    
    # Admin settings
    admin_api_key: Optional[str] = None
    enable_user_registration: bool = True


@dataclass
class User:
    """User model."""
    id: str
    username: str
    email: str
    api_key: str
    is_active: bool = True
    is_admin: bool = False
    created_at: str = ""
    last_login: Optional[str] = None
    usage_quota: int = 10000  # requests per month
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'api_key': self.api_key,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'usage_quota': self.usage_quota,
            'usage_count': self.usage_count
        }


class AuthManager:
    """Manage authentication and authorization."""
    
    def __init__(self, config: AuthConfig):
        """
        Initialize auth manager.
        
        Args:
            config: Authentication configuration
        """
        self.config = config
        
        # Initialize database
        self._init_database()
        
        # Initialize Redis for rate limiting
        self.redis_client = None
        if config.rate_limit_enabled:
            try:
                self.redis_client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Connected to Redis for rate limiting")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory rate limiting.")
                self.redis_client = None
                self.rate_limit_memory = defaultdict(list)
        
        # Security instances
        self.bearer_scheme = HTTPBearer(auto_error=False)
        self.api_key_header = APIKeyHeader(
            name=config.api_key_header_name,
            auto_error=False
        )
    
    def _init_database(self):
        """Initialize SQLite database for user management."""
        db_path = Path(self.config.auth_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                api_key_hash TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                is_admin BOOLEAN DEFAULT 0,
                created_at TEXT NOT NULL,
                last_login TEXT,
                usage_quota INTEGER DEFAULT 10000,
                usage_count INTEGER DEFAULT 0
            )
        """)
        
        # Create API key index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_key_hash 
            ON users (api_key_hash)
        """)
        
        # Create usage logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                response_time_ms REAL,
                status_code INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Create default admin user if needed
        if self.config.admin_api_key:
            self._create_admin_user()
    
    def _create_admin_user(self):
        """Create default admin user."""
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@tsukuyomi.local",
            api_key=self.config.admin_api_key,
            is_admin=True,
            created_at=datetime.now().isoformat()
        )
        
        try:
            self.create_user(admin_user)
            logger.info("Created admin user")
        except Exception as e:
            # Admin might already exist
            pass
    
    def create_user(self, user: User) -> User:
        """
        Create a new user.
        
        Args:
            user: User object
            
        Returns:
            Created user
        """
        conn = sqlite3.connect(self.config.auth_db_path)
        cursor = conn.cursor()
        
        # Generate API key if not provided
        if not user.api_key:
            user.api_key = self.generate_api_key()
        
        # Hash API key for storage
        api_key_hash = hashlib.sha256(user.api_key.encode()).hexdigest()
        
        # Set timestamps
        if not user.created_at:
            user.created_at = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT INTO users 
                (id, username, email, api_key, api_key_hash, is_active, 
                 is_admin, created_at, usage_quota)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.id, user.username, user.email, user.api_key,
                api_key_hash, user.is_active, user.is_admin,
                user.created_at, user.usage_quota
            ))
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.close()
            raise ValueError(f"User already exists: {e}")
        finally:
            conn.close()
        
        return user
    
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        conn = sqlite3.connect(self.config.auth_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, email, api_key, is_active, is_admin,
                   created_at, last_login, usage_quota, usage_count
            FROM users
            WHERE api_key_hash = ? AND is_active = 1
        """, (api_key_hash,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(
                id=row[0],
                username=row[1],
                email=row[2],
                api_key=row[3],
                is_active=bool(row[4]),
                is_admin=bool(row[5]),
                created_at=row[6],
                last_login=row[7],
                usage_quota=row[8],
                usage_count=row[9]
            )
        
        return None
    
    def authenticate_request(
        self,
        request: Request,
        api_key: Optional[str] = None,
        token: Optional[HTTPAuthorizationCredentials] = None
    ) -> User:
        """
        Authenticate a request.
        
        Args:
            request: FastAPI request
            api_key: API key from header
            token: JWT token
            
        Returns:
            Authenticated user
            
        Raises:
            HTTPException: If authentication fails
        """
        # Try API key authentication first
        if api_key:
            user = self.get_user_by_api_key(api_key)
            if user:
                self._update_last_login(user.id)
                return user
        
        # Try JWT token authentication
        if token and token.credentials:
            try:
                payload = jwt.decode(
                    token.credentials,
                    self.config.jwt_secret_key,
                    algorithms=[self.config.jwt_algorithm]
                )
                user_id = payload.get("user_id")
                if user_id:
                    user = self.get_user_by_id(user_id)
                    if user and user.is_active:
                        return user
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
        
        # No valid authentication found
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    def check_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            user_id: User ID or IP address
            endpoint: API endpoint
            
        Returns:
            True if within limit, False otherwise
        """
        if not self.config.rate_limit_enabled:
            return True
        
        key = f"rate_limit:{user_id}:{endpoint}"
        now = time.time()
        
        if self.redis_client:
            # Use Redis
            try:
                # Remove old entries
                self.redis_client.zremrangebyscore(
                    key, 0, now - self.config.rate_limit_window_seconds
                )
                
                # Count requests in window
                request_count = self.redis_client.zcard(key)
                
                if request_count >= self.config.rate_limit_max_requests:
                    return False
                
                # Add current request
                self.redis_client.zadd(key, {str(now): now})
                self.redis_client.expire(key, self.config.rate_limit_window_seconds + 1)
                
                return True
            except Exception as e:
                logger.error(f"Redis rate limit error: {e}")
                return True  # Fail open
        else:
            # Use in-memory storage
            # Clean old entries
            self.rate_limit_memory[key] = [
                t for t in self.rate_limit_memory[key]
                if t > now - self.config.rate_limit_window_seconds
            ]
            
            if len(self.rate_limit_memory[key]) >= self.config.rate_limit_max_requests:
                return False
            
            self.rate_limit_memory[key].append(now)
            return True
    
    def log_usage(
        self,
        user_id: str,
        endpoint: str,
        response_time_ms: float,
        status_code: int
    ):
        """Log API usage."""
        conn = sqlite3.connect(self.config.auth_db_path)
        cursor = conn.cursor()
        
        # Log usage
        cursor.execute("""
            INSERT INTO usage_logs 
            (user_id, endpoint, timestamp, response_time_ms, status_code)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id, endpoint, datetime.now().isoformat(),
            response_time_ms, status_code
        ))
        
        # Update usage count
        cursor.execute("""
            UPDATE users 
            SET usage_count = usage_count + 1
            WHERE id = ?
        """, (user_id,))
        
        conn.commit()
        conn.close()
    
    def check_usage_quota(self, user: User) -> bool:
        """Check if user is within usage quota."""
        return user.usage_count < user.usage_quota
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return f"tsk_{secrets.token_urlsafe(32)}"
    
    def generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "is_admin": user.is_admin,
            "exp": datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours)
        }
        
        return jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        conn = sqlite3.connect(self.config.auth_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, email, api_key, is_active, is_admin,
                   created_at, last_login, usage_quota, usage_count
            FROM users
            WHERE id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(
                id=row[0],
                username=row[1],
                email=row[2],
                api_key=row[3],
                is_active=bool(row[4]),
                is_admin=bool(row[5]),
                created_at=row[6],
                last_login=row[7],
                usage_quota=row[8],
                usage_count=row[9]
            )
        
        return None
    
    def _update_last_login(self, user_id: str):
        """Update user's last login time."""
        conn = sqlite3.connect(self.config.auth_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users
            SET last_login = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), user_id))
        
        conn.commit()
        conn.close()
    
    def get_usage_stats(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        conn = sqlite3.connect(self.config.auth_db_path)
        cursor = conn.cursor()
        
        # Get usage over time
        since = datetime.now() - timedelta(days=days)
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count,
                   AVG(response_time_ms) as avg_response_time
            FROM usage_logs
            WHERE user_id = ? AND timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (user_id, since.isoformat()))
        
        daily_usage = []
        for row in cursor.fetchall():
            daily_usage.append({
                'date': row[0],
                'count': row[1],
                'avg_response_time_ms': row[2]
            })
        
        # Get endpoint usage
        cursor.execute("""
            SELECT endpoint, COUNT(*) as count
            FROM usage_logs
            WHERE user_id = ? AND timestamp > ?
            GROUP BY endpoint
            ORDER BY count DESC
        """, (user_id, since.isoformat()))
        
        endpoint_usage = []
        for row in cursor.fetchall():
            endpoint_usage.append({
                'endpoint': row[0],
                'count': row[1]
            })
        
        conn.close()
        
        return {
            'daily_usage': daily_usage,
            'endpoint_usage': endpoint_usage,
            'total_requests': sum(d['count'] for d in daily_usage)
        }


# FastAPI dependencies
def get_auth_manager() -> AuthManager:
    """Get auth manager instance."""
    config = AuthConfig()
    return AuthManager(config)


async def authenticate(
    request: Request,
    api_key: Optional[str] = Security(APIKeyHeader(name="X-API-Key", auto_error=False)),
    token: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False)),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> User:
    """FastAPI dependency for authentication."""
    return auth_manager.authenticate_request(request, api_key, token)


async def require_admin(user: User = Depends(authenticate)) -> User:
    """FastAPI dependency for admin authentication."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


class RateLimitMiddleware:
    """Middleware for rate limiting."""
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
    
    async def __call__(self, request: Request, call_next):
        # Get user ID (from auth or IP)
        user_id = None
        
        # Try to get authenticated user
        try:
            api_key = request.headers.get("X-API-Key")
            if api_key:
                user = self.auth_manager.get_user_by_api_key(api_key)
                if user:
                    user_id = user.id
        except:
            pass
        
        # Fall back to IP address
        if not user_id:
            user_id = request.client.host
        
        # Check rate limit
        endpoint = request.url.path
        if not self.auth_manager.check_rate_limit(user_id, endpoint):
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log usage if authenticated
        if user_id and user_id != request.client.host:
            self.auth_manager.log_usage(
                user_id, endpoint, response_time_ms, response.status_code
            )
        
        return response