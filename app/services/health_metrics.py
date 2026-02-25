"""
Health Check and Metrics Service.

Provides comprehensive health checks and application metrics
for monitoring and observability.

Key features:
1. Liveness and readiness probes
2. Dependency health checks
3. Application metrics collection
4. Performance statistics
5. System resource monitoring
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyType(str, Enum):
    """Types of dependencies."""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    LLM_SERVICE = "llm_service"


@dataclass
class DependencyHealth:
    """Health status of a dependency."""
    name: str
    type: DependencyType
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "last_check": self.last_check.isoformat() if self.last_check else None
        }


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: Optional[float] = None
    disk_used_gb: Optional[float] = None
    disk_free_gb: Optional[float] = None
    open_files: int = 0
    threads: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "disk_percent": self.disk_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_free_gb": self.disk_free_gb,
            "open_files": self.open_files,
            "threads": self.threads
        }


@dataclass
class ApplicationMetrics:
    """Application-level metrics."""
    uptime_seconds: float
    requests_total: int
    requests_per_minute: float
    errors_total: int
    error_rate_percent: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    active_sessions: int = 0
    embeddings_generated: int = 0
    matches_created: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": self.uptime_seconds,
            "requests_total": self.requests_total,
            "requests_per_minute": self.requests_per_minute,
            "errors_total": self.errors_total,
            "error_rate_percent": self.error_rate_percent,
            "avg_response_time_ms": self.avg_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "active_sessions": self.active_sessions,
            "embeddings_generated": self.embeddings_generated,
            "matches_created": self.matches_created
        }


class HealthMetricsService:
    """
    Manages health checks and metrics collection.

    Tracks application health, dependencies, and performance metrics.
    """

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.version = os.getenv("APP_VERSION", "1.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")

        # Metrics tracking
        self._request_count = 0
        self._error_count = 0
        self._response_times: deque = deque(maxlen=1000)

        # Business metrics
        self._embeddings_generated = 0
        self._matches_created = 0
        self._active_sessions = 0

        # Dependency check cache
        self._dependency_cache: Dict[str, DependencyHealth] = {}
        self._cache_ttl_seconds = int(os.getenv("HEALTH_CACHE_TTL", "30"))

    def record_request(
        self,
        duration_ms: float,
        is_error: bool = False,
        endpoint: Optional[str] = None
    ) -> None:
        """Record a request for metrics."""
        self._request_count += 1
        if is_error:
            self._error_count += 1
        self._response_times.append(duration_ms)

    def record_embedding_generated(self) -> None:
        """Record an embedding generation."""
        self._embeddings_generated += 1

    def record_match_created(self) -> None:
        """Record a match creation."""
        self._matches_created += 1

    def update_active_sessions(self, count: int) -> None:
        """Update active session count."""
        self._active_sessions = count

    async def check_liveness(self) -> Dict[str, Any]:
        """
        Liveness check - is the application running?

        Returns minimal info for quick checks.
        """
        return {
            "status": HealthStatus.HEALTHY.value,
            "timestamp": datetime.utcnow().isoformat(),
            "version": self.version
        }

    async def check_readiness(self) -> Dict[str, Any]:
        """
        Readiness check - is the application ready to serve traffic?

        Checks critical dependencies.
        """
        dependencies = await self._check_all_dependencies()

        # Determine overall status
        all_healthy = all(d.status == HealthStatus.HEALTHY for d in dependencies)
        any_unhealthy = any(d.status == HealthStatus.UNHEALTHY for d in dependencies)

        if all_healthy:
            status = HealthStatus.HEALTHY
        elif any_unhealthy:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.DEGRADED

        return {
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "version": self.version,
            "environment": self.environment,
            "dependencies": [d.to_dict() for d in dependencies]
        }

    async def _check_all_dependencies(self) -> List[DependencyHealth]:
        """Check all dependencies."""
        dependencies = []

        # Database check
        db_health = await self._check_database()
        dependencies.append(db_health)

        # Cache check
        cache_health = await self._check_cache()
        dependencies.append(cache_health)

        return dependencies

    async def _check_database(self) -> DependencyHealth:
        """Check database connectivity."""
        cache_key = "database"
        cached = self._get_cached_check(cache_key)
        if cached:
            return cached

        start = time.perf_counter()
        try:
            # Try to import and check database
            from app.adapters.postgresql import postgresql_adapter
            if hasattr(postgresql_adapter, 'health_check'):
                result = postgresql_adapter.health_check()
            else:
                result = True  # Assume healthy if no health_check method

            latency = (time.perf_counter() - start) * 1000
            health = DependencyHealth(
                name="postgresql",
                type=DependencyType.DATABASE,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                latency_ms=round(latency, 2),
                last_check=datetime.utcnow()
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            health = DependencyHealth(
                name="postgresql",
                type=DependencyType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency, 2),
                message=str(e),
                last_check=datetime.utcnow()
            )

        self._cache_check(cache_key, health)
        return health

    async def _check_cache(self) -> DependencyHealth:
        """Check cache connectivity."""
        cache_key = "cache"
        cached = self._get_cached_check(cache_key)
        if cached:
            return cached

        start = time.perf_counter()
        try:
            from app.utils.cache import cache
            if hasattr(cache, 'ping'):
                result = cache.ping()
            else:
                result = True

            latency = (time.perf_counter() - start) * 1000
            health = DependencyHealth(
                name="redis",
                type=DependencyType.CACHE,
                status=HealthStatus.HEALTHY if result else HealthStatus.DEGRADED,
                latency_ms=round(latency, 2),
                last_check=datetime.utcnow()
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            health = DependencyHealth(
                name="redis",
                type=DependencyType.CACHE,
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 2),
                message=str(e),
                last_check=datetime.utcnow()
            )

        self._cache_check(cache_key, health)
        return health

    def _get_cached_check(self, key: str) -> Optional[DependencyHealth]:
        """Get cached health check if still valid."""
        if key not in self._dependency_cache:
            return None

        cached = self._dependency_cache[key]
        if not cached.last_check:
            return None

        age = datetime.utcnow() - cached.last_check
        if age.total_seconds() > self._cache_ttl_seconds:
            return None

        return cached

    def _cache_check(self, key: str, health: DependencyHealth) -> None:
        """Cache a health check result."""
        self._dependency_cache[key] = health

    def get_system_metrics(self) -> SystemMetrics:
        """Get system resource metrics."""
        try:
            import psutil
            process = psutil.Process()
            memory = psutil.virtual_memory()

            try:
                disk = psutil.disk_usage("/")
                disk_percent = disk.percent
                disk_used = round(disk.used / (1024 * 1024 * 1024), 2)
                disk_free = round(disk.free / (1024 * 1024 * 1024), 2)
            except Exception:
                disk_percent = None
                disk_used = None
                disk_free = None

            return SystemMetrics(
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=memory.percent,
                memory_used_mb=round(memory.used / (1024 * 1024), 2),
                memory_available_mb=round(memory.available / (1024 * 1024), 2),
                disk_percent=disk_percent,
                disk_used_gb=disk_used,
                disk_free_gb=disk_free,
                open_files=len(process.open_files()) if hasattr(process, 'open_files') else 0,
                threads=process.num_threads()
            )
        except ImportError:
            # psutil not available
            return SystemMetrics(
                cpu_percent=0,
                memory_percent=0,
                memory_used_mb=0,
                memory_available_mb=0
            )

    def get_application_metrics(self) -> ApplicationMetrics:
        """Get application metrics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        # Calculate requests per minute
        rpm = (self._request_count / uptime * 60) if uptime > 0 else 0

        # Calculate error rate
        error_rate = (self._error_count / self._request_count * 100) if self._request_count > 0 else 0

        # Calculate response time percentiles
        response_times = sorted(self._response_times)
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            p95_idx = int(len(response_times) * 0.95)
            p99_idx = int(len(response_times) * 0.99)
            p95_response = response_times[p95_idx] if p95_idx < len(response_times) else 0
            p99_response = response_times[p99_idx] if p99_idx < len(response_times) else 0
        else:
            avg_response = 0
            p95_response = 0
            p99_response = 0

        return ApplicationMetrics(
            uptime_seconds=round(uptime, 2),
            requests_total=self._request_count,
            requests_per_minute=round(rpm, 2),
            errors_total=self._error_count,
            error_rate_percent=round(error_rate, 2),
            avg_response_time_ms=round(avg_response, 2),
            p95_response_time_ms=round(p95_response, 2),
            p99_response_time_ms=round(p99_response, 2),
            active_sessions=self._active_sessions,
            embeddings_generated=self._embeddings_generated,
            matches_created=self._matches_created
        )

    async def get_full_metrics(self) -> Dict[str, Any]:
        """Get all metrics combined."""
        system = self.get_system_metrics()
        app = self.get_application_metrics()
        dependencies = await self._check_all_dependencies()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "version": self.version,
            "environment": self.environment,
            "system": system.to_dict(),
            "application": app.to_dict(),
            "dependencies": [d.to_dict() for d in dependencies]
        }

    def get_version_info(self) -> Dict[str, Any]:
        """Get version information."""
        return {
            "version": self.version,
            "environment": self.environment,
            "python_version": sys.version,
            "started_at": self.start_time.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
        }


# Global service instance
health_metrics_service = HealthMetricsService()
