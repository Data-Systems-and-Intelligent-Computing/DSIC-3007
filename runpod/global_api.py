"""Minimal Global Volume GraphQL calls from the pinned official Runpod SDK."""

import json
import subprocess

from runpod import screen

SOURCE = "https://github.com/runpod/runpod-python/tree/3864d374cba523ecee09f3bc95f073f17a64b970"


def graphql(query, variables=None):
    payload = json.dumps({"query": query, "variables": variables or {}})
    config = ('header = "Authorization: Bearer '+screen.api_key()+'"\n'
              'header = "Content-Type: application/json"\n'
              'data = "'+payload.replace('\\', '\\\\').replace('"', '\\"')+'"\n')
    result = subprocess.run(["curl", "--config", "-", "--silent", "--show-error",
                             "--max-time", "25", "https://api.runpod.io/graphql"],
                            input=config, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ConnectionError(f"Global Volume GraphQL client failed ({result.returncode}); no retry")
    response = json.loads(result.stdout)
    if response.get("errors"):
        # Messages may include input/env details. Persist only error codes.
        codes = [e.get("extensions", {}).get("validationErrorCode",
                  e.get("extensions", {}).get("code", "unspecified")) for e in response["errors"]]
        if codes and all(c == "GRAPHQL_VALIDATION_FAILED" for c in codes):
            # GraphQL validation precedes execution; this rejection cannot have
            # created a pod. Other failures remain uncertain and guarded.
            raise screen.HTTPFailure("POST", "GraphQL validation", 422)
        raise RuntimeError(f"Global Volume GraphQL rejected request: {codes}; no retry")
    if not isinstance(response.get("data"), dict):
        raise ValueError("GraphQL response lacks data; do not assume success")
    return response["data"]


def lookup_volume(config):
    data = graphql("query myGlobalVolumes { myself { globalStoreBuckets { id name } } }")
    rows = data["myself"]["globalStoreBuckets"]
    matches = [r for r in rows if r["id"] == config["global_volume_id"]]
    if len(matches) != 1 or matches[0]["name"] != config["global_volume_name"]:
        raise ValueError("selected Global Volume ID/name not confirmed; no create performed")
    return matches[0]


def deploy(payload):
    data = graphql("mutation deployPod($input: PodFindAndDeployOnDemandInput) { "
                   "podFindAndDeployOnDemand(input: $input) { id desiredStatus } }",
                   {"input": payload})
    pod = data.get("podFindAndDeployOnDemand")
    if not isinstance(pod, dict) or not pod.get("id"):
        raise ValueError("Global Volume create response lacks pod ID; ownership remains uncertain")
    return pod
