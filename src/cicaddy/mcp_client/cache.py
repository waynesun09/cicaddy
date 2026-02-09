"""Caching system for MCP client requests and responses."""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class CacheStats:
    """Statistics for cache performance tracking."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.errors = 0
        self.total_size = 0
        self.created_at = time.time()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        return (self.hits / total_requests) * 100.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate as percentage."""
        return 100.0 - self.hit_rate

    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1

    def record_set(self, size: int = 0):
        """Record a cache set operation."""
        self.sets += 1
        self.total_size += size

    def record_eviction(self, size: int = 0):
        """Record a cache eviction."""
        self.evictions += 1
        self.total_size -= size

    def record_error(self):
        """Record a cache error."""
        self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "total_size": self.total_size,
            "uptime": time.time() - self.created_at,
        }


class CacheEntry:
    """Cache entry with metadata."""

    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        size: Optional[int] = None,
    ):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.accessed_at = self.created_at
        self.access_count = 1
        self.ttl = ttl
        self.tags = tags or []
        self.size = size or len(str(value))

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at

    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass

    @abstractmethod
    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry by key."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get cache size."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction."""

    def __init__(
        self, max_size: int = 1000, max_memory: int = 100 * 1024 * 1024
    ):  # 100MB default
        self.max_size = max_size
        self.max_memory = max_memory
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU tracking
        self._current_memory = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired:
                await self._remove_entry(key)
                return None

            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            entry.touch()
            return entry

    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry."""
        async with self._lock:
            # Check if we need to evict
            await self._ensure_capacity(entry.size)

            # Remove existing entry if present
            if entry.key in self._cache:
                await self._remove_entry(entry.key)

            # Add new entry
            self._cache[entry.key] = entry
            self._access_order.append(entry.key)
            self._current_memory += entry.size

            return True

    async def delete(self, key: str) -> bool:
        """Delete cache entry by key."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
            return False

    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            self._current_memory = 0
            return count

    async def keys(self) -> List[str]:
        """Get all cache keys."""
        async with self._lock:
            return list(self._cache.keys())

    async def size(self) -> int:
        """Get cache size."""
        async with self._lock:
            return len(self._cache)

    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        # Remove expired entries first
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]
        for key in expired_keys:
            await self._remove_entry(key)

        # Check size limits
        while (
            len(self._cache) >= self.max_size
            or self._current_memory + new_entry_size > self.max_memory
        ):
            if not self._access_order:
                break
            # Remove least recently used entry
            lru_key = self._access_order[0]
            await self._remove_entry(lru_key)

    async def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            del self._cache[key]
            self._current_memory -= entry.size

        if key in self._access_order:
            self._access_order.remove(key)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "current_memory": self._current_memory,
            "max_memory": self.max_memory,
            "memory_usage_percent": (self._current_memory / self.max_memory) * 100,
            "entry_count": len(self._cache),
            "max_size": self.max_size,
            "size_usage_percent": (len(self._cache) / self.max_size) * 100,
        }


class MCPCache:
    """Main cache interface for MCP operations."""

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: float = 300.0,  # 5 minutes
        enable_compression: bool = False,
        namespace: str = "mcp",
    ):
        self.backend = backend or MemoryCache()
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.namespace = namespace
        self.stats = CacheStats()

    def _make_key(
        self, server_name: str, method: str, params: Optional[Dict] = None
    ) -> str:
        """Generate cache key for MCP request."""
        # Create deterministic key from request parameters
        key_data = {
            "server": server_name,
            "method": method,
            "params": params or {},
        }

        # Sort parameters for consistent key generation
        key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]

        return f"{self.namespace}:{server_name}:{method}:{key_hash}"

    async def get_cached_response(
        self, server_name: str, method: str, params: Optional[Dict] = None
    ) -> Optional[Any]:
        """
        Get cached response for MCP request.

        Args:
            server_name: Name of MCP server
            method: MCP method name
            params: Request parameters

        Returns:
            Cached response or None if not found
        """
        try:
            key = self._make_key(server_name, method, params)
            entry = await self.backend.get(key)

            if entry is None:
                self.stats.record_miss()
                logger.debug(f"Cache miss for {server_name}:{method}")
                return None

            self.stats.record_hit()
            logger.debug(
                f"Cache hit for {server_name}:{method} (age: {entry.age:.1f}s)"
            )
            return entry.value

        except Exception as e:
            self.stats.record_error()
            logger.warning(f"Cache get error for {server_name}:{method}: {e}")
            return None

    async def cache_response(
        self,
        server_name: str,
        method: str,
        params: Optional[Dict],
        response: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Cache response for MCP request.

        Args:
            server_name: Name of MCP server
            method: MCP method name
            params: Request parameters
            response: Response to cache
            ttl: Time to live in seconds
            tags: Tags for cache entry

        Returns:
            True if cached successfully
        """
        try:
            key = self._make_key(server_name, method, params)
            cache_ttl = ttl if ttl is not None else self.default_ttl

            # Don't cache error responses
            if isinstance(response, dict) and "error" in response:
                logger.debug(f"Not caching error response for {server_name}:{method}")
                return False

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=response,
                ttl=cache_ttl,
                tags=(tags or []) + [server_name, method],
            )

            success = await self.backend.set(entry)
            if success:
                self.stats.record_set(entry.size)
                logger.debug(
                    f"Cached response for {server_name}:{method} (ttl: {cache_ttl}s)"
                )
            return success

        except Exception as e:
            self.stats.record_error()
            logger.warning(f"Cache set error for {server_name}:{method}: {e}")
            return False

    async def invalidate(self, server_name: str, method: Optional[str] = None) -> int:
        """
        Invalidate cached entries.

        Args:
            server_name: Name of MCP server
            method: Specific method to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        try:
            keys = await self.backend.keys()
            pattern = f"{self.namespace}:{server_name}:"
            if method:
                pattern += f"{method}:"

            invalidated = 0
            for key in keys:
                if key.startswith(pattern):
                    if await self.backend.delete(key):
                        invalidated += 1

            logger.info(
                f"Invalidated {invalidated} cache entries for {server_name}:{method or 'all'}"
            )
            return invalidated

        except Exception as e:
            self.stats.record_error()
            logger.warning(f"Cache invalidation error for {server_name}: {e}")
            return 0

    async def clear_all(self) -> int:
        """Clear all cache entries."""
        try:
            count = await self.backend.clear()
            logger.info(f"Cleared {count} cache entries")
            return count
        except Exception as e:
            self.stats.record_error()
            logger.warning(f"Cache clear error: {e}")
            return 0

    def is_cacheable(self, method: str, params: Optional[Dict] = None) -> bool:
        """
        Determine if a request is cacheable.

        Args:
            method: MCP method name
            params: Request parameters

        Returns:
            True if request should be cached
        """
        # Don't cache write operations
        non_cacheable_methods = {
            "tools/call",  # Tool calls may have side effects
            "notifications/",  # Notifications are not cacheable
        }

        for non_cacheable in non_cacheable_methods:
            if method.startswith(non_cacheable):
                return False

        # Cache read operations
        cacheable_methods = {
            "tools/list",  # Tool listings are cacheable
            "resources/list",  # Resource listings are cacheable
            "prompts/list",  # Prompt listings are cacheable
        }

        return any(method.startswith(cacheable) for cacheable in cacheable_methods)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        base_stats = self.stats.get_stats()

        # Add backend-specific stats if available
        if hasattr(self.backend, "get_memory_usage"):
            base_stats.update(self.backend.get_memory_usage())

        return base_stats


class CacheMiddleware:
    """Middleware to add caching to MCP operations."""

    def __init__(self, cache: MCPCache):
        self.cache = cache

    async def execute_with_cache(
        self,
        server_name: str,
        method: str,
        params: Optional[Dict],
        operation_func,
        force_refresh: bool = False,
        custom_ttl: Optional[float] = None,
    ) -> Any:
        """
        Execute operation with caching.

        Args:
            server_name: Name of MCP server
            method: MCP method name
            params: Request parameters
            operation_func: Function to execute if cache miss
            force_refresh: Force refresh cache
            custom_ttl: Custom TTL for this request

        Returns:
            Operation result
        """
        # Check if request is cacheable
        if not self.cache.is_cacheable(method, params):
            logger.debug(f"Method {method} is not cacheable, executing directly")
            return await operation_func()

        # Try cache first (unless force refresh)
        if not force_refresh:
            cached_response = await self.cache.get_cached_response(
                server_name, method, params
            )
            if cached_response is not None:
                return cached_response

        # Execute operation
        logger.debug(f"Executing {server_name}:{method} (cache miss or force refresh)")
        response = await operation_func()

        # Cache the response
        await self.cache.cache_response(
            server_name=server_name,
            method=method,
            params=params,
            response=response,
            ttl=custom_ttl,
        )

        return response


# Global cache instance
default_cache = MCPCache()
cache_middleware = CacheMiddleware(default_cache)
