import logging
from typing import Dict, Any, Optional, List, Tuple

from open_webui.llm.utils.nemo_guardrails.service import NemoGuardrailService

logger = logging.getLogger(__name__)

_service = NemoGuardrailService()


class Guardrail:
    """Guardrail service wrapper using embedded NemoGuardrails (no external HTTP calls)."""

    def __init__(self, backend_host: Optional[str] = None):
        self._service = _service

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def list_guardrails(
        self, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List all NIM guardrail configs."""
        return await self._service.list_guardrails(params)

    async def get_guardrail(self, guardrail_id: str, org_id: str = "") -> Optional[Dict[str, Any]]:
        """Get a single guardrail config by ID."""
        return await self._service.get_guardrail(guardrail_id, org_id)

    async def create_guardrail(
        self, data: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
        """Create a new guardrail config.

        Returns:
            Tuple of (result, error_status, error_detail)
        """
        return await self._service.create_guardrail(data)

    async def update_guardrail(
        self, guardrail_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update an existing guardrail config."""
        return await self._service.update_guardrail(guardrail_id, data)

    async def delete_guardrail(self, guardrail_id: str, org_id: str = "") -> bool:
        """Delete a guardrail config by ID."""
        return await self._service.delete_guardrail(guardrail_id, org_id)

    async def check_safety(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check message safety using NemoGuardrails."""
        return await self._service.check_safety(data)


# Factory function (kept for backward compatibility)
def create_guardrail_service(backend_host: Optional[str] = None) -> Guardrail:
    return Guardrail(backend_host)


# Module-level async functions for convenience
async def list_guardrails(
    params: Optional[Dict[str, Any]] = None, backend_host: Optional[str] = None
) -> List[Dict[str, Any]]:
    async with Guardrail(backend_host) as service:
        return await service.list_guardrails(params)


async def get_guardrail(
    guardrail_id: str, backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with Guardrail(backend_host) as service:
        return await service.get_guardrail(guardrail_id)


async def create_guardrail(
    data: Dict[str, Any], backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with Guardrail(backend_host) as service:
        return await service.create_guardrail(data)


async def update_guardrail(
    guardrail_id: str, data: Dict[str, Any], backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with Guardrail(backend_host) as service:
        return await service.update_guardrail(guardrail_id, data)


async def delete_guardrail(
    guardrail_id: str, backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with Guardrail(backend_host) as service:
        return await service.delete_guardrail(guardrail_id)


async def check_safety(
    data: Dict[str, Any], backend_host: Optional[str] = None
) -> Dict[str, Any]:
    async with Guardrail(backend_host) as service:
        return await service.check_safety(data)
