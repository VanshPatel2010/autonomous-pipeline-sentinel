"""MemoryStore: three-tier long-term memory for the autonomous repair agent.

Architectural Rationale
-----------------------
The current system stores repair outcomes in a flat ``repair_memory`` table
and retrieves them by exact anomaly-type match.  This is fundamentally
limited: two incidents with different ``anomaly_type`` labels but the same
root cause are treated as completely unrelated.

This module introduces three complementary memory tiers that together let
the Planner reason like an experienced engineer who remembers *what happened*,
*what it means*, and *how to fix it*:

1. **Episodic Memory** — "What happened?"
   Specific repair events with full context.  Retrieved by semantic similarity
   of root-cause text, enabling cross-type retrieval.  Example: "replica lag
   after failover" matches previous "network latency" incidents.

2. **Semantic Memory** — "What does this mean?"
   Abstract general knowledge distilled from many episodes.  Example:
   "replica_lag_after_failover usually resolves in 3-5 minutes if the network
   is stable".  Confidence is updated as evidence accumulates.

3. **Procedural Memory** — "How do I fix this?"
   Successful repair step-sequences stored as reusable procedures.
   The Planner prefers procedures with high success_rate over templates.

Design decisions
----------------
- All three tiers are backed by SQLite tables so no additional infrastructure
  is required to run the agent.
- Each retrieval method returns a typed list for easy consumption.
- Semantic similarity is currently keyword-overlap cosine approximation.
  The ``SemanticRetriever`` interface is stable; swapping in a vector DB
  (Chroma, Pinecone, Weaviate, FAISS) requires replacing one method.
- The MemoryStore is write-optimised: every repair cycle writes one entry
  to each tier.  Reads are fast because tables are indexed.

Future extensibility
--------------------
- Vector DB: replace ``_keyword_similarity`` with embedding lookup.
- Knowledge Graph: replace SemanticMemory with an RDF triple store.
- Reinforcement Learning: expose episodic memory as replay buffer.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from db.client import get_db_connection

from logging_config import logger

from agents.repair.models import (
    EpisodicMemoryEntry,
    ProceduralMemoryEntry,
    RepairContext,
    SemanticMemoryEntry,
    SelfReflection,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Semantic similarity (keyword-based, vector-DB-ready interface)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set:
    """Tokenise a text string into a set of lowercase words."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two text strings.

    This is the default semantic similarity implementation.
    To upgrade to embedding-based similarity, replace this function
    with a call to an embedding model + cosine similarity computation.

    Args:
        text_a: First text string.
        text_b: Second text string.

    Returns:
        Jaccard similarity in [0.0, 1.0].
    """
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


class SemanticRetriever:
    """Interface for semantic similarity retrieval.

    This class wraps the similarity function so that replacing
    keyword-based similarity with a vector DB is a one-class change.

    Future implementations
    ----------------------
    - ``ChromaRetriever``: embed text with sentence-transformers, store in Chroma.
    - ``PineconeRetriever``: use Pinecone's managed vector index.
    - ``FAISSRetriever``: local FAISS index for offline deployment.
    """

    def similarity(self, query: str, candidate: str) -> float:
        """Return a similarity score in [0.0, 1.0] between query and candidate."""
        return _keyword_similarity(query, candidate)

    def rank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        text_key: str,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Rank a list of candidate dicts by similarity to the query.

        Args:
            query      : Query text.
            candidates : List of dicts to rank.
            text_key   : Key in each dict whose value is the comparison text.
            top_k      : Return at most this many results.
            min_score  : Minimum similarity to include a result.

        Returns:
            List of (candidate_dict, score) tuples, sorted by score desc.
        """
        scored = [
            (c, self.similarity(query, str(c.get(text_key, ""))))
            for c in candidates
        ]
        scored = [(c, s) for c, s in scored if s >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """Three-tier long-term memory: Episodic, Semantic, Procedural.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (same file as the rest of the system).
    retriever : SemanticRetriever, optional
        Injected similarity retriever.  Defaults to keyword-Jaccard.

    Usage::

        store = MemoryStore(db_path=db_path)

        # Write episode after each repair
        store.write_episode(ctx, verification, strategy_name, recovery_secs)

        # Read similar episodes before planning
        episodes = store.retrieve_similar_episodes(ctx.root_cause, top_k=5)

        # Update procedural memory on success
        store.update_procedure(strategy_name, ctx, plan_steps, verification)
    """

    def __init__(
        self,
        db_path: str,
        retriever: Optional[SemanticRetriever] = None,
    ) -> None:
        self.db_path = db_path
        self._retriever = retriever or SemanticRetriever()

    # ------------------------------------------------------------------
    # Episodic Memory
    # ------------------------------------------------------------------

    def write_episode(
        self,
        ctx: RepairContext,
        verification: VerificationResult,
        strategy_name: str,
        recovery_secs: float,
    ) -> str:
        """Write a repair event to episodic memory.

        The embedding_text is crafted from the most semantically rich
        fields so that future retrieval captures root-cause similarity
        rather than just anomaly-type exact match.

        Args:
            ctx           : RepairContext snapshot.
            verification  : Post-repair VerificationResult.
            strategy_name : Strategy that was executed.
            recovery_secs : Actual recovery time in seconds.

        Returns:
            The memory_id of the created entry.
        """
        outcome = "success" if verification.passed else (
            "partial" if verification.verification_score > 0.4 else "failure"
        )

        # Rich embedding text for semantic retrieval
        embedding_text = (
            f"{ctx.anomaly_type} {ctx.root_cause} {ctx.severity} "
            f"{ctx.active_node_label} {' '.join(ctx.affected_tables)} "
            f"{strategy_name} {outcome}"
        )

        entry = EpisodicMemoryEntry(
            run_id=ctx.run_id,
            anomaly_type=ctx.anomaly_type,
            severity=ctx.severity,
            root_cause=ctx.root_cause,
            strategy_used=strategy_name,
            outcome=outcome,
            verification_score=verification.verification_score,
            recovery_secs=recovery_secs,
            embedding_text=embedding_text,
            embedding=[],  # populated by future vector encoder
        )

        try:
            conn = get_db_connection()
            conn.execute(
                """
                INSERT INTO episodic_memory (
                    memory_id, run_id, anomaly_type, severity, root_cause,
                    strategy_used, outcome, verification_score, recovery_secs,
                    embedding_text, created_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """ if conn.is_postgres else 
                """
                INSERT INTO episodic_memory (
                    memory_id, run_id, anomaly_type, severity, root_cause,
                    strategy_used, outcome, verification_score, recovery_secs,
                    embedding_text, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    entry.memory_id, entry.run_id, entry.anomaly_type,
                    entry.severity, entry.root_cause, entry.strategy_used,
                    entry.outcome, entry.verification_score, entry.recovery_secs,
                    entry.embedding_text, entry.created_at,
                ),
            )
            conn.commit()
            conn.close()
            logger.debug(
                f"[{ctx.run_id}] MemoryStore: episodic entry written "
                f"(id={entry.memory_id}, outcome={outcome})"
            )
            return entry.memory_id

        except Exception as exc:
            logger.error(f"[{ctx.run_id}] MemoryStore.write_episode: {exc}")
            return ""

    def retrieve_similar_episodes(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Retrieve episodic memories semantically similar to the query.

        Uses the SemanticRetriever to rank stored episodes by text
        similarity to the query.  The query should be constructed from
        the current root_cause, anomaly_type, and affected system.

        This is the key method designed for vector DB replacement:
        swap ``_retriever.rank()`` with an embedding lookup.

        Args:
            query_text : Text describing the current incident.
            top_k      : Maximum number of results to return.
            min_score  : Minimum similarity score to include.

        Returns:
            List of episode dicts sorted by similarity, each with
            an additional ``similarity_score`` field.
        """
        try:
            conn = get_db_connection()
            rows = conn.execute(
                """
                SELECT * FROM episodic_memory
                ORDER BY created_at DESC LIMIT 200
                """
            ).fetchall()
            conn.close()

            candidates = [dict(r) for r in rows]
            ranked = self._retriever.rank(
                query=query_text,
                candidates=candidates,
                text_key="embedding_text",
                top_k=top_k,
                min_score=min_score,
            )
            results = []
            for doc, score in ranked:
                doc["similarity_score"] = round(score, 4)
                results.append(doc)

            logger.debug(
                f"MemoryStore: retrieved {len(results)} similar episodes "
                f"for query='{query_text[:60]}...'"
            )
            return results

        except Exception as exc:
            logger.warning(f"MemoryStore.retrieve_similar_episodes: {exc}")
            return []

    # ------------------------------------------------------------------
    # Semantic Memory
    # ------------------------------------------------------------------

    def write_semantic_knowledge(
        self,
        concept: str,
        knowledge: str,
        confidence: float = 0.6,
        source: str = "learner",
    ) -> None:
        """Upsert a general knowledge statement into semantic memory.

        If a statement with the same concept exists, its confidence is
        updated using a Bayesian-style weighted blend. If not, a new
        entry is created.

        Args:
            concept    : Short concept identifier (snake_case).
            knowledge  : Natural-language knowledge statement.
            confidence : Initial confidence for this statement [0,1].
            source     : Origin of this knowledge ('learner'|'llm'|'operator').
        """
        try:
            conn = get_db_connection()
            existing = conn.execute(
                "SELECT * FROM semantic_memory WHERE concept = %s" if conn.is_postgres else
                "SELECT * FROM semantic_memory WHERE concept = ?",
                (concept,),
            ).fetchone()

            now = datetime.now(timezone.utc).isoformat()

            if existing:
                # Bayesian blend: 70% existing + 30% new evidence
                new_conf = 0.70 * existing["confidence"] + 0.30 * confidence
                conn.execute(
                    """
                    UPDATE semantic_memory
                    SET knowledge=%s, confidence=%s, updated_at=%s
                    WHERE concept=%s
                    """ if conn.is_postgres else
                    """
                    UPDATE semantic_memory
                    SET knowledge=?, confidence=?, updated_at=?
                    WHERE concept=?
                    """,
                    (knowledge, round(new_conf, 4), now, concept),
                )
            else:
                import uuid as _uuid
                mem_id = str(_uuid.uuid4())[:12]
                conn.execute(
                    """
                    INSERT INTO semantic_memory
                        (memory_id, concept, knowledge, confidence, source, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """ if conn.is_postgres else
                    """
                    INSERT INTO semantic_memory
                        (memory_id, concept, knowledge, confidence, source, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (mem_id, concept, knowledge, round(confidence, 4), source, now, now),
                )

            conn.commit()
            conn.close()

        except Exception as exc:
            logger.warning(f"MemoryStore.write_semantic_knowledge: {exc}")

    def retrieve_relevant_knowledge(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> List[SemanticMemoryEntry]:
        """Retrieve semantic knowledge entries relevant to the query.

        Args:
            query_text : Current incident description.
            top_k      : Maximum entries to return.

        Returns:
            List of SemanticMemoryEntry objects sorted by confidence × relevance.
        """
        try:
            conn = get_db_connection()
            rows = conn.execute("SELECT * FROM semantic_memory").fetchall()
            conn.close()

            candidates = [dict(r) for r in rows]
            ranked = self._retriever.rank(
                query=query_text,
                candidates=candidates,
                text_key="knowledge",
                top_k=top_k,
                min_score=0.05,
            )

            results = []
            for row, _ in ranked:
                results.append(SemanticMemoryEntry(
                    memory_id=row.get("memory_id", ""),
                    concept=row.get("concept", ""),
                    knowledge=row.get("knowledge", ""),
                    confidence=row.get("confidence", 0.5),
                    source=row.get("source", ""),
                    created_at=row.get("created_at", ""),
                    updated_at=row.get("updated_at", ""),
                ))
            return results

        except Exception as exc:
            logger.warning(f"MemoryStore.retrieve_relevant_knowledge: {exc}")
            return []

    # ------------------------------------------------------------------
    # Procedural Memory
    # ------------------------------------------------------------------

    def update_procedure(
        self,
        strategy_name: str,
        ctx: RepairContext,
        plan_steps: List[Dict[str, Any]],
        verification: VerificationResult,
    ) -> None:
        """Update or create a procedural memory entry for a repair strategy.

        Successful repairs increment success_count and improve the
        average verification score.  Failed repairs only increment
        total_count, reducing the procedure's apparent success rate.
        This implements adaptive confidence (Requirement 7).

        Args:
            strategy_name : Strategy that was executed.
            ctx           : RepairContext.
            plan_steps    : Serialised list of executed steps.
            verification  : Post-repair VerificationResult.
        """
        success = verification.passed

        try:
            conn = get_db_connection()
            existing = conn.execute(
                """
                SELECT * FROM procedural_memory
                WHERE procedure_name=%s AND anomaly_type=%s AND severity=%s
                """ if conn.is_postgres else
                """
                SELECT * FROM procedural_memory
                WHERE procedure_name=? AND anomaly_type=? AND severity=?
                """,
                (strategy_name, ctx.anomaly_type, ctx.severity),
            ).fetchone()

            now = datetime.now(timezone.utc).isoformat()
            steps_json = json.dumps(plan_steps)

            if existing:
                new_total = existing["total_count"] + 1
                new_success = existing["success_count"] + (1 if success else 0)
                # Rolling average of verification score
                old_avg = existing["avg_verification_score"]
                new_avg = (old_avg * existing["total_count"] + verification.verification_score) / new_total

                conn.execute(
                    """
                    UPDATE procedural_memory
                    SET success_count=%s, total_count=%s, avg_verification_score=%s,
                        steps=%s, updated_at=%s
                    WHERE procedure_name=%s AND anomaly_type=%s AND severity=%s
                    """ if conn.is_postgres else
                    """
                    UPDATE procedural_memory
                    SET success_count=?, total_count=?, avg_verification_score=?,
                        steps=?, updated_at=?
                    WHERE procedure_name=? AND anomaly_type=? AND severity=?
                    """,
                    (
                        new_success, new_total, round(new_avg, 4),
                        steps_json, now,
                        strategy_name, ctx.anomaly_type, ctx.severity,
                    ),
                )
            else:
                import uuid as _uuid
                mem_id = str(_uuid.uuid4())[:12]
                conn.execute(
                    """
                    INSERT INTO procedural_memory (
                        memory_id, procedure_name, anomaly_type, severity,
                        steps, success_count, total_count, avg_verification_score,
                        created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """ if conn.is_postgres else
                    """
                    INSERT INTO procedural_memory (
                        memory_id, procedure_name, anomaly_type, severity,
                        steps, success_count, total_count, avg_verification_score,
                        created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        mem_id, strategy_name, ctx.anomaly_type, ctx.severity,
                        steps_json, 1 if success else 0, 1,
                        round(verification.verification_score, 4),
                        now, now,
                    ),
                )

            conn.commit()
            conn.close()

            logger.debug(
                f"[{ctx.run_id}] MemoryStore: procedural entry updated "
                f"({strategy_name} | success={success})"
            )

        except Exception as exc:
            logger.error(f"[{ctx.run_id}] MemoryStore.update_procedure: {exc}")

    def get_best_procedure(
        self,
        anomaly_type: str,
        severity: str,
        min_trials: int = 3,
    ) -> Optional[ProceduralMemoryEntry]:
        """Return the highest-success-rate procedure for this anomaly+severity.

        Procedures with fewer than ``min_trials`` attempts are excluded
        to avoid being fooled by lucky one-off successes.

        Args:
            anomaly_type : Anomaly type to filter.
            severity     : Severity level to filter.
            min_trials   : Minimum number of attempts required.

        Returns:
            Best ProceduralMemoryEntry or None.
        """
        try:
            conn = get_db_connection()
            rows = conn.execute(
                """
                SELECT * FROM procedural_memory
                WHERE anomaly_type=%s AND severity=%s AND total_count >= %s
                ORDER BY (success_count * 1.0 / total_count) DESC
                LIMIT 1
                """ if conn.is_postgres else
                """
                SELECT * FROM procedural_memory
                WHERE anomaly_type=? AND severity=? AND total_count >= ?
                ORDER BY (success_count * 1.0 / total_count) DESC
                LIMIT 1
                """,
                (anomaly_type, severity, min_trials),
            ).fetchall()
            conn.close()

            if not rows:
                return None

            r = dict(rows[0])
            try:
                steps = json.loads(r.get("steps", "[]"))
            except json.JSONDecodeError:
                steps = []

            return ProceduralMemoryEntry(
                memory_id=r.get("memory_id", ""),
                procedure_name=r.get("procedure_name", ""),
                anomaly_type=r.get("anomaly_type", ""),
                severity=r.get("severity", ""),
                steps=steps,
                success_count=r.get("success_count", 0),
                total_count=r.get("total_count", 0),
                avg_verification_score=r.get("avg_verification_score", 0.0),
                created_at=r.get("created_at", ""),
                updated_at=r.get("updated_at", ""),
            )

        except Exception as exc:
            logger.warning(f"MemoryStore.get_best_procedure: {exc}")
            return None

    # ------------------------------------------------------------------
    # Self-Reflection Storage (Requirement 9)
    # ------------------------------------------------------------------

    def write_reflection(self, reflection: SelfReflection) -> None:
        """Persist a SelfReflection entry to the database.

        Args:
            reflection : SelfReflection dataclass from the RepairerAgent.
        """
        try:
            conn = get_db_connection()
            conn.execute(
                """
                INSERT INTO self_reflections (
                    run_id, plan_id, strategy_chosen, verification_passed,
                    verification_score, optimal_choice, cheaper_alternative,
                    downtime_reduction, confidence_adjustment,
                    learning_points, reflection_text, created_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """ if conn.is_postgres else
                """
                INSERT INTO self_reflections (
                    run_id, plan_id, strategy_chosen, verification_passed,
                    verification_score, optimal_choice, cheaper_alternative,
                    downtime_reduction, confidence_adjustment,
                    learning_points, reflection_text, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    reflection.run_id,
                    reflection.plan_id,
                    reflection.strategy_chosen,
                    int(reflection.verification_passed),
                    reflection.verification_score,
                    int(reflection.optimal_choice),
                    reflection.cheaper_alternative or "",
                    reflection.downtime_reduction,
                    reflection.confidence_adjustment,
                    json.dumps(reflection.learning_points),
                    reflection.reflection_text,
                    reflection.created_at,
                ),
            )
            conn.commit()
            conn.close()
            logger.debug(
                f"[{reflection.run_id}] MemoryStore: self-reflection stored "
                f"(plan={reflection.plan_id})"
            )

        except Exception as exc:
            logger.error(f"MemoryStore.write_reflection: {exc}")

    def get_recent_reflections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent self-reflections.

        Args:
            limit : Maximum number of entries to return.

        Returns:
            List of reflection dicts, newest first.
        """
        try:
            conn = get_db_connection()
            rows = conn.execute(
                """
                SELECT * FROM self_reflections
                ORDER BY created_at DESC LIMIT %s
                """ if conn.is_postgres else
                """
                SELECT * FROM self_reflections
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

        except Exception as exc:
            logger.warning(f"MemoryStore.get_recent_reflections: {exc}")
            return []

    # ------------------------------------------------------------------
    # Adaptive confidence query
    # ------------------------------------------------------------------

    def get_adaptive_confidence(
        self,
        strategy_name: str,
        anomaly_type: str,
        severity: str,
    ) -> Tuple[float, int]:
        """Compute adaptive confidence for a strategy from all memory tiers.

        Combines:
        - Procedural memory success rate (strongest signal).
        - Episodic memory recent outcomes (recency-weighted).

        Args:
            strategy_name : Strategy to evaluate.
            anomaly_type  : Current anomaly type.
            severity      : Current severity.

        Returns:
            Tuple of (confidence: float, sample_count: int).
        """
        # 1. Procedural memory (broad)
        proc = self.get_best_procedure(anomaly_type, severity, min_trials=2)
        proc_conf = proc.success_rate if proc and proc.procedure_name == strategy_name else None

        # 2. Episodic memory (recent)
        query = f"{strategy_name} {anomaly_type} {severity}"
        episodes = self.retrieve_similar_episodes(query, top_k=10, min_score=0.1)
        recent = [
            e for e in episodes
            if e.get("strategy_used") == strategy_name
        ]
        recent_success = sum(1 for e in recent if e.get("outcome") == "success")
        recent_conf = recent_success / len(recent) if recent else None

        # Blend
        if proc_conf is not None and recent_conf is not None:
            confidence = 0.6 * proc_conf + 0.4 * recent_conf
        elif proc_conf is not None:
            confidence = proc_conf
        elif recent_conf is not None:
            confidence = recent_conf
        else:
            confidence = 0.5  # no data

        sample_count = (proc.total_count if proc else 0) + len(recent)
        return round(confidence, 4), sample_count

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------


