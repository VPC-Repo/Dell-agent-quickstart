"""Run policies and an idempotent queue of immutable snapshots."""

from __future__ import annotations

import hashlib
import json

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from . import service, store


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RunPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timeout_seconds: int = Field(default=600, ge=1, le=3600)
    max_attempts: int = Field(default=1, ge=1, le=3)
    token_budget: int = Field(default=0, ge=0, le=1000000)


def enqueue(
    snapshot, query, ref, policy=None, conversation_key="", idempotency_key="", **kwargs
):
    policy = RunPolicy.model_validate(policy or {}).model_dump()
    execution = {
        "policy": policy,
        "conversation_key": conversation_key,
        **kwargs.pop("execution", {}),
    }
    snapshot = {**snapshot, "execution": execution}
    signature = digest({"snapshot": snapshot, "query": query, "ref": ref})
    snapshot = {**snapshot, "request_sha256": signature}
    occurrence = (
        "request:" + idempotency_key
        if idempotency_key
        else kwargs.pop("occurrence", None)
    )
    run = service.make_run(snapshot, query, ref, occurrence=occurrence, **kwargs)
    try:
        with store.Session.begin() as db:
            db.add(run)
    except IntegrityError:
        if not idempotency_key:
            raise
        with store.Session() as db:
            prior = db.scalar(
                select(store.Run).where(store.Run.occurrence == occurrence)
            )
            if not prior:
                raise
            prior_signature = prior.snapshot.get("request_sha256") or digest(
                {
                    "snapshot": prior.snapshot,
                    "query": prior.query,
                    "ref": prior.blueprint_ref,
                }
            )
            if prior_signature != signature:
                raise HTTPException(
                    409, "Idempotency key was already used with different input"
                ) from None
            return store.serialize(prior)
    return store.serialize(run)
