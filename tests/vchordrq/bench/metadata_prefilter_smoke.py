#!/usr/bin/env python3
"""Synthetic smoke bench for the vchordrq metadata prefilter.

This is intentionally small enough for ad hoc regression checks. It expects a
database with the vchord extension installed and the Python `psycopg` package
available.

Shape notes (do not casually change):

* `VECTOR_DIM = 128`. The metadata prefilter saves wall-clock by skipping
  vector arithmetic on rejected candidates inside the vchord scan. At
  `vector(3)` the distance compute is trivially cheap, so even when the index
  is used the speedup degrades to ~1.2x and gets lost in noise.

* `SET enable_seqscan = off` in `run_timed`. On a 100k-row synthetic table the
  cost-fix branch's planner correctly prices Seq Scan as cheaper than the
  vchord index scan, which would bypass the feature under test entirely (both
  modes execute the same Seq Scan plan and produce 1.0x "speedup"). Forcing
  the index is the supported way to exercise the metadata-prefilter code
  path in a regression test.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import time

import psycopg


VECTOR_DIM = 128

QUERY = f"""
SELECT id
FROM metadata_prefilter_smoke
WHERE tenant_hash = hashtextextended('feed-10', 0)
  AND state_code = 0
  AND deletion_marker = 0
ORDER BY v <-> ('[' || array_to_string(array_fill(0.13::real, ARRAY[{VECTOR_DIM}]), ',') || ']')::vector
LIMIT 50
"""

DEBUG_NOTICE_PREFIX = "vchordrq_metadata_prefilter "
QUAL_DIAGNOSTICS_PREFIX = "vchordrq_metadata_qual_diagnostics "

GENERIC_PARAM_QUERY = f"""
SELECT id
FROM metadata_prefilter_smoke
WHERE tenant_hash = hashtextextended('feed-10', 0)
  AND state_code = %(state)s::bigint
  AND deletion_marker = %(deleted)s::bigint
  AND (flag_bits & %(mask)s::bigint) = %(mask)s::bigint
  AND geo_token = ANY(%(geo)s::bigint[])
ORDER BY v <-> ('[' || array_to_string(array_fill(0.13::real, ARRAY[{VECTOR_DIM}]), ',') || ']')::vector
LIMIT 50
"""

GENERIC_PREPARED_QUERY = f"""
SELECT id
FROM metadata_prefilter_smoke
WHERE tenant_hash = hashtextextended('feed-10', 0)
  AND state_code = $1::bigint
  AND deletion_marker = $2::bigint
  AND (flag_bits & $3::bigint) = $3::bigint
  AND geo_token = ANY($4::bigint[])
ORDER BY v <-> ('[' || array_to_string(array_fill(0.13::real, ARRAY[{VECTOR_DIM}]), ',') || ']')::vector
LIMIT 50
"""

GENERIC_EXECUTE_ARGS = (
    "0::bigint, 0::bigint, 3::bigint, ARRAY[10,20,30,40,50]::bigint[]"
)


def configure_prefilter(cur: psycopg.Cursor, mode: str, debug: bool = False) -> None:
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET vchordrq.prefilter = on")
    cur.execute(f"SET vchordrq.metadata_prefilter = {mode}")
    cur.execute(
        "SET vchordrq.metadata_active_columns = "
        "'tenant_hash,state_code,deletion_marker'"
    )
    cur.execute(f"SET vchordrq.metadata_prefilter_debug = {'on' if debug else 'off'}")


def assert_vchord_plan(cur: psycopg.Cursor, mode: str, context: str) -> None:
    configure_prefilter(cur, mode)
    cur.execute("SET enable_sort = on")
    cur.execute(f"EXPLAIN (FORMAT TEXT, COSTS OFF) {QUERY}")
    plan = "\n".join(row[0] for row in cur.fetchall())
    if "metadata_prefilter_smoke_idx" not in plan:
        raise AssertionError(
            f"planner did not choose vchordrq metadata index {context}:\n" + plan
        )


def run_timed(
    cur: psycopg.Cursor, mode: str, debug: bool, repeats: int, notices: list[str]
) -> tuple[list[int], float]:
    configure_prefilter(cur, mode, debug)

    notices.clear()
    timings = []
    result: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter()
        cur.execute(QUERY)
        result = [row[0] for row in cur.fetchall()]
        timings.append(time.perf_counter() - started)
    return result, statistics.median(timings)


def parse_counters(notice_body: str) -> dict[str, str]:
    """Pull `key=value` pairs out of a vchordrq_metadata_prefilter NOTICE line."""
    return dict(re.findall(r"(\w+)=([^ ]+)", notice_body))


def assert_relation_stats(cur: psycopg.Cursor, *rel_names: str) -> None:
    cur.execute(
        """
        SELECT relname, reltuples
        FROM pg_class
        WHERE relname = ANY(%s)
        """,
        (list(rel_names),),
    )
    stats = {relname: float(reltuples) for relname, reltuples in cur.fetchall()}
    missing = [relname for relname in rel_names if stats.get(relname, 0.0) <= 0.0]
    if missing:
        raise AssertionError(f"ANALYZE left zero reltuples for {missing}: {stats}")


def top_total_cost(plan: str) -> float:
    match = re.search(r"\.\.([0-9]+(?:\.[0-9]+)?) rows=", plan)
    if match is None:
        raise AssertionError("could not parse top-level total cost from plan:\n" + plan)
    return float(match.group(1))


def assert_generic_param_metadata_quals(
    cur: psycopg.Cursor, notices: list[str]
) -> None:
    cur.execute("SET client_min_messages = log")
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute("SET plan_cache_mode = force_generic_plan")
    cur.execute("SET vchordrq.prefilter = on")
    cur.execute("SET vchordrq.metadata_prefilter = reject_only")
    cur.execute(
        "SET vchordrq.metadata_active_columns = "
        "'tenant_hash,state_code,deletion_marker,flag_bits,geo_token'"
    )
    cur.execute("SET vchordrq.metadata_qual_diagnostics = on")

    notices.clear()
    cur.execute(
        GENERIC_PARAM_QUERY,
        {
            "state": 0,
            "deleted": 0,
            "mask": 3,
            "geo": [10, 20, 30, 40, 50],
        },
        prepare=True,
    )
    cur.fetchall()
    diagnostics = [n for n in notices if n.startswith(QUAL_DIAGNOSTICS_PREFIX)]
    if not diagnostics:
        raise AssertionError("expected vchordrq metadata qual diagnostics NOTICE")
    counters = parse_counters(diagnostics[-1])
    expected_columns = {
        "tenant_hash",
        "state_code",
        "deletion_marker",
        "flag_bits",
        "geo_token",
    }
    detected_columns = set(counters.get("metadata_detected_columns", "").split(","))
    if counters.get("metadata_supported_qual_count") != "5":
        raise AssertionError(f"generic params should support all quals: {diagnostics[-1]!r}")
    if counters.get("metadata_unsupported_qual_count") != "0":
        raise AssertionError(f"generic params should not leave unsupported quals: {diagnostics[-1]!r}")
    if counters.get("metadata_unavailable_param_count") != "0":
        raise AssertionError(f"generic params should resolve once per scan: {diagnostics[-1]!r}")
    if not expected_columns.issubset(detected_columns):
        raise AssertionError(f"missing generic-param metadata columns: {diagnostics[-1]!r}")

    cur.execute("SET vchordrq.metadata_qual_diagnostics = off")
    cur.execute("SET plan_cache_mode = auto")
    cur.execute("RESET client_min_messages")


def assert_prepared_generic_plan_cost(cur: psycopg.Cursor) -> None:
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute("SET plan_cache_mode = force_generic_plan")
    cur.execute("SET vchordrq.prefilter = on")
    cur.execute(
        "SET vchordrq.metadata_active_columns = "
        "'tenant_hash,state_code,deletion_marker,flag_bits,geo_token'"
    )

    costs: dict[str, float] = {}
    plans: dict[str, str] = {}
    for mode in ("off", "reject_only"):
        name = f"metadata_prefilter_smoke_generic_{mode}"
        cur.execute(f"SET vchordrq.metadata_prefilter = {mode}")
        cur.execute(
            f"PREPARE {name}(bigint, bigint, bigint, bigint[]) "
            f"AS {GENERIC_PREPARED_QUERY}"
        )
        cur.execute(
            f"EXPLAIN (FORMAT TEXT, COSTS ON) EXECUTE {name}"
            f"({GENERIC_EXECUTE_ARGS})"
        )
        plan = "\n".join(row[0] for row in cur.fetchall())
        plans[mode] = plan
        costs[mode] = top_total_cost(plan)
        cur.execute(f"DEALLOCATE {name}")

    if "metadata_prefilter_smoke_idx" not in plans["reject_only"]:
        raise AssertionError(
            "prepared generic reject_only plan did not choose vchordrq index:\n"
            + plans["reject_only"]
        )
    if costs["reject_only"] >= costs["off"]:
        raise AssertionError(
            "metadata prefilter should lower generic prepared vchord cost: "
            f"off={costs['off']:.2f} reject_only={costs['reject_only']:.2f}\n"
            f"off plan:\n{plans['off']}\nreject_only plan:\n{plans['reject_only']}"
        )

    cur.execute("SET plan_cache_mode = auto")
    cur.execute("RESET enable_bitmapscan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL", ""))
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    args = parser.parse_args()

    dsn = args.dsn or f"dbname={os.environ['USER']}"
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        notices: list[str] = []
        conn.add_notice_handler(lambda diag: notices.append(diag.message_primary))
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vchord")
            cur.execute("DROP TABLE IF EXISTS metadata_prefilter_smoke")
            cur.execute(
                f"""
                CREATE TABLE metadata_prefilter_smoke (
                  id int PRIMARY KEY,
                  tenant_hash bigint,
                  state_code bigint,
                  visibility_code bigint,
                  deletion_marker bigint,
                  flag_bits bigint,
                  geo_token bigint,
                  time_bucket bigint,
                  v vector({VECTOR_DIM}) NOT NULL
                )
                """
            )
            cur.execute(
                f"""
                INSERT INTO metadata_prefilter_smoke
                SELECT i,
                       hashtextextended('feed-' || (i %% 100)::text, 0),
                       (i %% 5)::bigint,
                       (i %% 2)::bigint,
                       ((i %% 97) = 0)::int::bigint,
                       CASE WHEN i %% 8 = 0 THEN 7 ELSE 3 END,
                       (i %% 512)::bigint,
                       (i / 256)::bigint,
                       (
                         SELECT array_agg(
                           ((i::bigint * 1103515245 + j::bigint * 12345) %% 65536)::real / 65536.0
                         )::real[]::vector
                         FROM generate_series(1, {VECTOR_DIM}) j
                       )
                FROM generate_series(1, %s) i
                """,
                (args.rows,),
            )
            cur.execute(
                """
                CREATE INDEX metadata_prefilter_smoke_idx
                ON metadata_prefilter_smoke
                USING vchordrq (v vector_l2_ops)
                INCLUDE (
                  tenant_hash,
                  state_code,
                  visibility_code,
                  deletion_marker,
                  flag_bits,
                  geo_token,
                  time_bucket
                )
                WITH (options = $$
                residual_quantization = false
                rerank_in_table = false
                [metadata]
                columns = [
                  { name = "tenant_hash", ops = ["eq", "in"], exact = false },
                  { name = "state_code", ops = ["eq", "in"], exact = true },
                  { name = "visibility_code", ops = ["eq"], exact = true },
                  { name = "deletion_marker", ops = ["eq"], exact = true },
                  { name = "flag_bits", ops = ["eq", "bitmask_contains"], exact = true },
                  { name = "geo_token", ops = ["eq", "in", "range"], exact = false },
                  { name = "time_bucket", ops = ["range"], exact = false },
                ]
                [build.internal]
                lists = []
                $$)
                """
            )
            cur.execute("ANALYZE metadata_prefilter_smoke")
            assert_relation_stats(
                cur, "metadata_prefilter_smoke", "metadata_prefilter_smoke_idx"
            )
            cur.execute(
                "CREATE INDEX metadata_prefilter_smoke_filter_idx "
                "ON metadata_prefilter_smoke (tenant_hash, state_code, deletion_marker)"
            )
            cur.execute("ANALYZE metadata_prefilter_smoke")
            assert_relation_stats(
                cur,
                "metadata_prefilter_smoke",
                "metadata_prefilter_smoke_idx",
                "metadata_prefilter_smoke_filter_idx",
            )

            assert_vchord_plan(
                cur,
                "reject_only",
                "with competing btree metadata index and sort enabled",
            )

            # The competing btree exists only for the planner-shape assertion
            # above. Timings must compare metadata modes on the same vchord
            # access path, not a btree+sort shortcut in one mode.
            cur.execute("DROP INDEX metadata_prefilter_smoke_filter_idx")
            cur.execute("ANALYZE metadata_prefilter_smoke")
            assert_relation_stats(
                cur, "metadata_prefilter_smoke", "metadata_prefilter_smoke_idx"
            )
            assert_vchord_plan(cur, "off", "for timed off-mode run")
            assert_vchord_plan(cur, "reject_only", "for timed reject_only run")
            assert_generic_param_metadata_quals(cur, notices)
            assert_prepared_generic_plan_cost(cur)

            off_result, off_s = run_timed(cur, "off", False, args.repeats, notices)
            if any(n.startswith(DEBUG_NOTICE_PREFIX) for n in notices):
                raise AssertionError(
                    "metadata_prefilter_debug=off must not emit any vchordrq NOTICE"
                )

            reject_result, reject_s = run_timed(
                cur, "reject_only", False, args.repeats, notices
            )
            if any(n.startswith(DEBUG_NOTICE_PREFIX) for n in notices):
                raise AssertionError(
                    "metadata_prefilter_debug=off must not emit any vchordrq NOTICE "
                    "even when metadata_prefilter=reject_only"
                )

            debug_result, debug_s = run_timed(
                cur, "reject_only", True, args.repeats, notices
            )
            debug_notices = [n for n in notices if n.startswith(DEBUG_NOTICE_PREFIX)]
            if len(debug_notices) != args.repeats:
                raise AssertionError(
                    f"expected one vchordrq_metadata_prefilter NOTICE per repeat "
                    f"(got {len(debug_notices)} for {args.repeats} repeats)"
                )
            counters = parse_counters(debug_notices[0])
            for required in (
                "metadata_checked",
                "metadata_rejected",
                "metadata_rejected_by_columns",
                "metadata_false_negative_debug",
                "metadata_prefilter_mode",
            ):
                if required not in counters:
                    raise AssertionError(
                        f"NOTICE is missing expected counter `{required}`: "
                        f"{debug_notices[0]!r}"
                    )
            if counters["metadata_prefilter_mode"] != "reject_only":
                raise AssertionError(
                    f"NOTICE reports mode={counters['metadata_prefilter_mode']!r}, "
                    f"expected reject_only"
                )
            if int(counters["metadata_rejected"]) <= 0:
                raise AssertionError(
                    f"reject_only mode should have rejected candidates: {debug_notices[0]!r}"
                )
            if counters["metadata_rejected_by_columns"] == "none":
                raise AssertionError(
                    f"reject_only mode should report reject columns: {debug_notices[0]!r}"
                )
            if int(counters["metadata_false_negative_debug"]) != 0:
                raise AssertionError(
                    f"debug verification surfaced a false-negative reject: "
                    f"{debug_notices[0]!r}"
                )

            if off_result != reject_result or reject_result != debug_result:
                raise AssertionError("metadata prefilter changed query results")

            cur.execute(
                """
                SELECT count(*)
                FROM metadata_prefilter_smoke
                WHERE NOT (
                  tenant_hash = hashtextextended('feed-10', 0)
                  AND state_code = 0
                  AND deletion_marker = 0
                )
                """
            )
            logical_rejects = cur.fetchone()[0]
            if logical_rejects <= 0:
                raise AssertionError("synthetic predicate did not reject any rows")

            speedup = off_s / reject_s if reject_s > 0 else float("inf")
            print(
                f"off={off_s:.4f}s reject_only={reject_s:.4f}s "
                f"debug={debug_s:.4f}s speedup={speedup:.2f}x "
                f"logical_rejects={logical_rejects}"
            )
            if speedup < args.min_speedup:
                raise AssertionError(
                    f"metadata prefilter speedup {speedup:.2f}x < {args.min_speedup:.2f}x"
                )


if __name__ == "__main__":
    main()
