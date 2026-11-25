"""LLM / tool / HTTP counters and latencies (OpenTelemetry metrics)."""

from __future__ import annotations

from opentelemetry import metrics

from gateway.config.settings import Settings

_meter = metrics.get_meter("gateway")

llm_calls = _meter.create_counter("gateway.llm.calls", description="Model turns")
llm_latency = _meter.create_histogram("gateway.llm.latency", unit="s")
llm_tokens = _meter.create_counter("gateway.llm.tokens", description="Tokens by direction")
tool_calls = _meter.create_counter("gateway.tool.calls", description="Tool calls by outcome")
tool_latency = _meter.create_histogram("gateway.tool.latency", unit="s")
http_requests = _meter.create_counter("gateway.http.requests")
http_latency = _meter.create_histogram("gateway.http.latency", unit="s")
runs = _meter.create_counter("gateway.runs", description="Runs by final status")


def setup_metrics(settings: Settings) -> None:
    if not settings.otel_exporter_otlp_endpoint:
        return
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    base = settings.otel_exporter_otlp_endpoint.rstrip("/")
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{base}/v1/metrics"))
    metrics.set_meter_provider(
        MeterProvider(
            resource=Resource.create({"service.name": settings.service_name}),
            metric_readers=[reader],
        )
    )
