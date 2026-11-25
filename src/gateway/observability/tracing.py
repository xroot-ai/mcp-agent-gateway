"""OpenTelemetry tracing. Langfuse is fed through its OTLP endpoint, so one set of spans
serves both Langfuse and any OTel backend."""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

from gateway.config.settings import Settings

log = logging.getLogger(__name__)
_tracer = trace.get_tracer("gateway")


def setup_tracing(settings: Settings) -> TracerProvider | None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    exporters = []
    if settings.otel_exporter_otlp_endpoint:
        base = settings.otel_exporter_otlp_endpoint.rstrip("/")
        exporters.append(OTLPSpanExporter(endpoint=f"{base}/v1/traces"))
    if settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key:
        creds = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        exporters.append(
            OTLPSpanExporter(
                endpoint=f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
                headers={"Authorization": "Basic " + base64.b64encode(creds).decode()},
            )
        )
    if not exporters:
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    for exporter in exporters:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    log.info("tracing enabled (%d exporter(s))", len(exporters))
    return provider


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    attrs = {k: v for k, v in attributes.items() if isinstance(v, str | int | float | bool)}
    with _tracer.start_as_current_span(name, attributes=attrs) as s:
        yield s


def mark_interrupted(s: Span, reason: str) -> None:
    """Flag a span whose work was cut short (cancel, timeout, block) so dashboards can
    tell an interrupted run from a failed one."""
    s.set_attribute("run.interrupted", True)
    s.set_attribute("run.interrupt_reason", reason)
    s.set_status(Status(StatusCode.ERROR, f"interrupted: {reason}"))
