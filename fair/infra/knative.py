"""KNative service management for live model serving.

Public URLs are recorded on each base-model STAC item as the
`mlm:inference-endpoint` asset. Consumers read STAC; they do not build URLs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pystac

KNATIVE_GROUP = "serving.knative.dev"
KNATIVE_VERSION = "v1"
KNATIVE_PLURAL = "services"
DEFAULT_NAMESPACE = "predict"
S3_CREDENTIALS_SECRET = "s3-credentials"


def knative_service_name(name: str) -> str:
    """Convert a model identifier to a DNS-1035 label accepted by KNative."""
    return str(name).lower().replace("_", "-")


def knative_service_host(name: str, namespace: str = DEFAULT_NAMESPACE) -> str:
    service_name = knative_service_name(name)
    return f"{service_name}.{namespace}.svc.cluster.local"


def public_predict_url(name: str, domain: str) -> str:
    # Must stay in lock-step with config-domain, domain-template, and the
    # wildcard Ingress; this is the only place the shape lives.
    return f"https://{knative_service_name(name)}.predict.{domain}/predict"


def _module_from_entrypoint(entrypoint: str) -> str:
    if ":" not in entrypoint:
        msg = f"Invalid mlm:entrypoint '{entrypoint}', expected 'module.path:function'"
        raise ValueError(msg)
    return entrypoint.rsplit(":", 1)[0]


def _service_name(item: pystac.Item) -> str:
    return knative_service_name(item.properties.get("mlm:name") or item.id)


def build_knative_manifest(item: pystac.Item, namespace: str = DEFAULT_NAMESPACE) -> dict[str, Any]:
    inference_asset = item.assets.get("mlm:inference")
    if inference_asset is None:
        msg = f"Item '{item.id}' missing 'mlm:inference' asset"
        raise KeyError(msg)

    source_asset = item.assets.get("source-code")
    if source_asset is None:
        msg = f"Item '{item.id}' missing 'source-code' asset"
        raise KeyError(msg)
    entrypoint = source_asset.extra_fields.get("mlm:entrypoint")
    if not entrypoint:
        msg = f"Item '{item.id}' source-code asset missing 'mlm:entrypoint'"
        raise KeyError(msg)

    return {
        "apiVersion": f"{KNATIVE_GROUP}/{KNATIVE_VERSION}",
        "kind": "Service",
        "metadata": {
            "name": _service_name(item),
            "namespace": namespace,
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "autoscaling.knative.dev/min-scale": "0",
                        "autoscaling.knative.dev/max-scale": "5",
                        "autoscaling.knative.dev/scale-down-delay": "60s",
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "image": inference_asset.href,
                            "ports": [{"containerPort": 8080}],
                            "env": [
                                {"name": "MODEL_MODULE", "value": _module_from_entrypoint(entrypoint)},
                            ],
                            "envFrom": [
                                {"secretRef": {"name": S3_CREDENTIALS_SECRET}},
                            ],
                        }
                    ],
                },
            }
        },
    }


def _custom_objects_api() -> Any:
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CustomObjectsApi()


def _upsert_resource(
    *,
    read: Callable[[], Any],
    create: Callable[[], Any],
    patch: Callable[[], Any],
) -> None:
    from kubernetes.client.exceptions import ApiException

    try:
        read()
    except ApiException as exc:
        if exc.status != 404:
            raise
        create()
        return

    patch()


def _upsert_knative_service(api: Any, manifest: dict[str, Any], namespace: str) -> None:
    name = manifest["metadata"]["name"]
    _upsert_resource(
        read=lambda: api.get_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            name=name,
        ),
        create=lambda: api.create_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            body=manifest,
        ),
        patch=lambda: api.patch_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            name=name,
            body=manifest,
        ),
    )


def ensure_knative_service(item: pystac.Item, namespace: str = DEFAULT_NAMESPACE) -> None:
    if not _knative_serving_installed():
        print(f"skip knative: {KNATIVE_GROUP}/{KNATIVE_VERSION} not registered on cluster")
        return
    manifest = build_knative_manifest(item, namespace=namespace)
    api = _custom_objects_api()

    _upsert_knative_service(api, manifest, namespace)


def _knative_serving_installed() -> bool:
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    try:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        groups = client.ApisApi().get_api_versions().groups
    except (config.ConfigException, ApiException):
        return False
    return any(g.name == KNATIVE_GROUP for g in groups)


def delete_knative_service(model_name: str, namespace: str = DEFAULT_NAMESPACE) -> None:
    from kubernetes.client.exceptions import ApiException

    api = _custom_objects_api()
    name = knative_service_name(model_name)
    try:
        api.delete_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            name=name,
        )
    except ApiException as exc:
        if exc.status != 404:
            raise
