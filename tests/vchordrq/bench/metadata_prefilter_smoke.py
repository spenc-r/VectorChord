#!/usr/bin/env python3
"""Synthetic smoke bench for the vchordrq metadata prefilter.

This is intentionally small enough for ad hoc regression checks. It expects a
database with the vchord extension installed and the Python `psycopg` package
available.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import psycopg


QUERY = """
SELECT id
FROM metadata_prefilter_smoke
WHERE feed_id_meta_hash = hashtextextended('feed-17', 0)
  AND status_meta = 1
  AND deleted_meta = 0
ORDER BY v <-> '[0.13,0.21,0.34]'::vector
LIMIT 50
"""


def assert_vchord_plan(cur: psycopg.Cursor) -> None:
    cur.execute("SET enable_sort = on")
    cur.execute("SET vchordrq.prefilter = on")
    cur.execute("SET vchordrq.metadata_prefilter = 'reject_only'")
    cur.execute(
        "SET vchordrq.metadata_active_columns = "
        "'feed,flags,status,deleted,visibility,geo,time'"
    )
    cur.execute(f"EXPLAIN (FORMAT TEXT, COSTS OFF) {QUERY}")
    plan = "\n".join(row[0] for row in cur.fetchall())
    if "metadata_prefilter_smoke_idx" not in plan:
        raise AssertionError(
            "planner did not choose vchordrq metadata index with sort enabled:\n" + plan
        )


def run_timed(cur: psycopg.Cursor, mode: str, debug: bool, repeats: int) -> tuple[list[int], float]:
    cur.execute("SET vchordrq.prefilter = on")
    cur.execute("SET vchordrq.metadata_prefilter = %s", (mode,))
    cur.execute(
        "SET vchordrq.metadata_active_columns = "
        "'feed,flags,status,deleted,visibility,geo,time'"
    )
    cur.execute("SET vchordrq.metadata_prefilter_debug = %s", ("on" if debug else "off",))

    timings = []
    result: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter()
        cur.execute(QUERY)
        result = [row[0] for row in cur.fetchall()]
        timings.append(time.perf_counter() - started)
    return result, statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL", ""))
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    args = parser.parse_args()

    with psycopg.connect(args.dsn or f"dbname={os.environ['USER']}") as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vchord")
            cur.execute("DROP TABLE IF EXISTS metadata_prefilter_smoke")
            cur.execute(
                """
                CREATE TABLE metadata_prefilter_smoke (
                  id int PRIMARY KEY,
                  feed_id_meta_hash bigint,
                  status_meta bigint,
                  visibility_meta bigint,
                  deleted_meta bigint,
                  eligibility_flags_meta bigint,
                  geo_cell_meta bigint,
                  created_at_bucket_meta bigint,
                  v vector(3) NOT NULL
                )
                """
            )
            cur.execute(
                """
                INSERT INTO metadata_prefilter_smoke
                SELECT i,
                       hashtextextended('feed-' || (i % 100)::text, 0),
                       (i % 5)::bigint,
                       (i % 2)::bigint,
                       ((i % 97) = 0)::int::bigint,
                       CASE WHEN i % 8 = 0 THEN 7 ELSE 3 END,
                       (i % 512)::bigint,
                       (i / 256)::bigint,
                       ARRAY[
                         (i % 997) / 997.0,
                         (i % 991) / 991.0,
                         (i % 983) / 983.0
                       ]::real[]::vector
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
                  feed_id_meta_hash,
                  status_meta,
                  visibility_meta,
                  deleted_meta,
                  eligibility_flags_meta,
                  geo_cell_meta,
                  created_at_bucket_meta
                )
                WITH (options = $$
                residual_quantization = false
                rerank_in_table = false
                [build.internal]
                lists = []
                $$)
                """
            )
            cur.execute("ANALYZE metadata_prefilter_smoke")
            cur.execute(
                "CREATE INDEX metadata_prefilter_smoke_filter_idx "
                "ON metadata_prefilter_smoke (feed_id_meta_hash, status_meta, deleted_meta)"
            )
            cur.execute("ANALYZE metadata_prefilter_smoke")

            assert_vchord_plan(cur)

            off_result, off_s = run_timed(cur, "off", False, args.repeats)
            reject_result, reject_s = run_timed(cur, "reject_only", False, args.repeats)
            debug_result, debug_s = run_timed(cur, "reject_only", True, args.repeats)

            if off_result != reject_result or reject_result != debug_result:
                raise AssertionError("metadata prefilter changed query results")

            cur.execute(
                """
                SELECT count(*)
                FROM metadata_prefilter_smoke
                WHERE NOT (
                  feed_id_meta_hash = hashtextextended('feed-17', 0)
                  AND status_meta = 1
                  AND deleted_meta = 0
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
