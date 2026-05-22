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
import statistics
import time

import psycopg


VECTOR_DIM = 128

QUERY = f"""
SELECT id
FROM metadata_prefilter_smoke
WHERE feed_id_meta_hash = hashtextextended('feed-10', 0)
  AND status_meta = 0
  AND deleted_meta = 0
ORDER BY v <-> ('[' || array_to_string(array_fill(0.13::real, ARRAY[{VECTOR_DIM}]), ',') || ']')::vector
LIMIT 50
"""


def run_timed(cur: psycopg.Cursor, mode: str, debug: bool, repeats: int) -> tuple[list[int], float]:
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET vchordrq.prefilter = on")
    cur.execute(f"SET vchordrq.metadata_prefilter = {mode}")
    cur.execute(
        "SET vchordrq.metadata_active_columns = "
        "'feed,flags,status,deleted,visibility,geo,time'"
    )
    cur.execute(f"SET vchordrq.metadata_prefilter_debug = {'on' if debug else 'off'}")

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
                f"""
                CREATE TABLE metadata_prefilter_smoke (
                  id int PRIMARY KEY,
                  feed_id_meta_hash bigint,
                  status_meta bigint,
                  visibility_meta bigint,
                  deleted_meta bigint,
                  eligibility_flags_meta bigint,
                  geo_cell_meta bigint,
                  created_at_bucket_meta bigint,
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
                  feed_id_meta_hash = hashtextextended('feed-10', 0)
                  AND status_meta = 0
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
