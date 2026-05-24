# vchordrq Index-Resident Metadata Prefilter

This page describes the `vchordrq` index-resident metadata prefilter available on this branch.

The feature lets a `vchordrq` index store a small set of bigint metadata values beside each vector candidate. During an ANN scan, VectorChord can inspect those resident values and reject candidates that cannot satisfy selected scalar predicates before fetching the heap row. This is useful for queries such as "nearest vectors for this tenant", "nearest vectors in these buckets", or "nearest vectors whose denormalized flags contain this mask".

The metadata prefilter is a candidate filter, not a replacement for PostgreSQL query semantics. Keep the original heap predicates in your SQL. Metadata predicates should be conservative accelerators that reject impossible matches and let possible matches continue to the normal heap recheck.

## When To Use It

Use index-resident metadata when your vector searches usually include selective scalar predicates and the vector index is still the best access path:

```sql
SELECT id, embedding
FROM items
WHERE tenant_id = 'acme'
  AND status = 'active'
ORDER BY embedding <-> $1::vector
LIMIT 100;
```

Without metadata predicates, `vchordrq` may have to fetch many heap rows before finding enough rows for the tenant/status filter. With resident metadata, the index can discard obvious non-matches while it is still scanning ANN candidates.

This feature is most helpful when:

- the query has an `ORDER BY vector_distance LIMIT ...`;
- filters are common and moderately selective;
- filter evaluation or heap fetches are expensive;
- scalar B-tree, GiST, or GIN indexes are not clearly better for that query.

It is less useful for very tight scalar filters where a normal scalar index plus sort is cheaper, or for filters that cannot be represented as bigint metadata.

## How It Works

There are three pieces:

1. Add bigint metadata columns to the table.
2. Include those columns in the `vchordrq` index and declare their supported operations in the index `options`.
3. Add matching metadata predicates to the query, alongside the original predicates.

For example, a plain tenant query might start as:

```sql
SELECT id
FROM items
WHERE tenant_id = 'acme'
ORDER BY embedding <-> $1::vector
LIMIT 100;
```

With a generated hash metadata column, the accelerated query becomes:

```sql
SELECT id
FROM items
WHERE tenant_hash = hashtextextended('acme', 0)
  AND tenant_id = 'acme'
ORDER BY embedding <-> $1::vector
LIMIT 100;
```

`tenant_hash = ...` is the resident metadata predicate. `tenant_id = ...` is still the authoritative heap predicate. The index does not infer the hash from `tenant_id = ...`; your application or SQL must provide the metadata predicate.

During the index scan, each candidate carries the stored `tenant_hash` value. If the value is present and does not match the query hash, the candidate is rejected before heap prefiltering. If the value matches, is missing, or cannot prove rejection, the candidate continues to the ordinary heap recheck.

## Table Columns

Metadata columns must be `bigint` columns included in the same `vchordrq` index. A generated stored column is strongly recommended so inserts and updates cannot leave metadata stale:

```sql
CREATE TABLE items (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL,
  status text,
  updated_at timestamptz,
  embedding vector(768) NOT NULL,

  tenant_hash bigint
    GENERATED ALWAYS AS (hashtextextended(tenant_id, 0)) STORED,

  status_hash bigint
    GENERATED ALWAYS AS (
      CASE WHEN status IS NULL THEN NULL
           ELSE hashtextextended(lower(status), 0)
      END
    ) STORED,

  updated_day bigint
    GENERATED ALWAYS AS (
      CASE WHEN updated_at IS NULL THEN NULL
           ELSE floor(extract(epoch FROM updated_at) / 86400.0)::bigint
      END
    ) STORED
);
```

If you maintain metadata columns yourself, every insert and update path must keep them consistent with the real columns. Stale metadata can cause false negatives when `metadata_prefilter = reject_only`.

## Index Syntax

Declare metadata columns with normal PostgreSQL `INCLUDE (...)` syntax and a `[metadata]` block in the `vchordrq` options:

```sql
CREATE INDEX items_embedding_idx
ON items
USING vchordrq (embedding vector_l2_ops)
INCLUDE (tenant_hash, status_hash, updated_day)
WITH (options = $$
residual_quantization = false
rerank_in_table = false

[metadata]
columns = [
  { name = "tenant_hash", ops = ["eq", "in"], exact = false },
  { name = "status_hash", ops = ["eq", "in"], exact = false },
  { name = "updated_day", ops = ["eq", "in", "range"], exact = false },
]

[build.internal]
lists = []
$$);
```

Rules:

- the index must have exactly one vector key column;
- each declared metadata column must be a `bigint` `INCLUDE` column;
- up to 32 metadata INCLUDE columns are supported;
- names in `[metadata].columns` must be unique;
- `ops` controls which SQL predicate shapes are recognized;
- `exact` controls whether the metadata value can ever replace the heap prefilter in `covered_skip_heap` mode.

## Supported Operations

`ops` is a list of operation names:

| Operation | SQL shape | Example |
|---|---|---|
| `eq` | equality | `tenant_hash = $2::bigint` |
| `in` | scalar array membership | `tenant_hash = ANY($2::bigint[])` |
| `range` | bigint comparisons | `updated_day >= $2::bigint` |
| `bitmask_contains` | mask containment | `(flags_meta & $2::bigint) = $2::bigint` |

The right-hand side may be a constant or a supported prepared-statement parameter. Integer widening casts are accepted, for example `$1::bigint`.

For prepared statements, bind the metadata value itself as an integer parameter. The metadata detector does not evaluate arbitrary expressions such as `hashtextextended($text_param, 0)` or `floor(extract(epoch FROM $ts_param) / 86400.0)` during scan setup. Those expressions may be constant-folded when all inputs are SQL literals, but parameterized applications should compute or fetch the metadata value before executing the vector query.

Unsupported predicates are ignored by the metadata prefilter and remain normal heap predicates. A query is still correct if no metadata predicate is detected; it simply gets no resident-metadata speedup.

## Query Syntax

Queries should contain both metadata predicates and original predicates. In prepared statements, pass precomputed metadata values as separate bigint parameters.

Hashed string:

```sql
SELECT id
FROM items
WHERE tenant_hash = $3::bigint     -- hashtextextended($2, 0)
  AND tenant_id = $2
ORDER BY embedding <-> $1::vector
LIMIT 100;
```

Hashed normalized string:

```sql
SELECT id
FROM items
WHERE status_hash = $3::bigint     -- hashtextextended(lower($2), 0)
  AND lower(status) = lower($2)
ORDER BY embedding <-> $1::vector
LIMIT 100;
```

Bucketed time range:

```sql
SELECT id
FROM items
WHERE updated_day >= $3::bigint    -- floor(extract(epoch FROM $2) / 86400.0)
  AND updated_at >= $2::timestamptz
ORDER BY embedding <-> $1::vector
LIMIT 100;
```

Bitmask:

```sql
SELECT id
FROM items
WHERE (flags_meta & $2::bigint) = $2::bigint
  AND expensive_original_filter(...)
ORDER BY embedding <-> $1::vector
LIMIT 100;
```

The metadata predicate should be conservative. It must never reject a row that could pass the original predicate.

## Runtime Settings

The metadata prefilter is controlled by GUCs:

```sql
SET vchordrq.prefilter = on;
SET vchordrq.metadata_prefilter = reject_only;
SET vchordrq.metadata_active_columns = '';
```

`vchordrq.metadata_prefilter` accepts:

- `off`: do not evaluate index-resident metadata.
- `reject_only`: use metadata only to reject candidates that definitely cannot match. This is the recommended default.
- `covered_skip_heap`: if every scan predicate is covered by exact metadata, skip the AM-side heap prefilter. Use only with metadata encodings that are truly exact.

`vchordrq.metadata_active_columns` is a comma-separated list of declared metadata column names. Empty, `all`, or `*` activates all declared metadata columns:

```sql
SET vchordrq.metadata_active_columns = 'tenant_hash,status_hash';
SET vchordrq.metadata_active_columns = 'all';
SET vchordrq.metadata_active_columns = '';
```

Use actual metadata column names, not aliases. Unknown names are ignored.

`vchordrq.metadata_block_prune` enables experimental block-level metadata pruning. Leave it off unless you are specifically testing that path.

`vchordrq.metadata_prefilter_debug` verifies metadata rejections against the heap prefilter and raises an error on a false negative. It also emits per-scan instrumentation notices. This is useful in staging, but it adds overhead.

`vchordrq.metadata_qual_diagnostics` emits diagnostics about which scan predicates were detected as metadata predicates. The debug and diagnostics GUCs are superuser-set.

## Exactness

The `exact` flag is about whether metadata can be trusted as the full predicate, not whether the metadata value is useful for rejection.

Use `exact = false` for lossy or conservative encodings:

- hashes of text values;
- geo cells that over-approximate `ST_DWithin`;
- day buckets for timestamp predicates;
- bloom or bitmask encodings with possible collisions.

Use `exact = true` only for lossless encodings:

- a `bigint` column copied directly into a metadata column;
- a boolean encoded as `0` or `1`;
- a small enum encoded bijectively as `bigint`;
- an exact bitmask where `(flags & mask) = mask` is the real predicate.

In `reject_only` mode, `exact = false` is still useful. The metadata can reject definite non-matches and defer possible matches to the heap recheck.

In `covered_skip_heap` mode, only exact, fully covered predicates may skip the AM-side heap prefilter. If you are unsure, use `reject_only`.

## NULLs And Missing Metadata

NULL metadata values are not stored as valid resident metadata values. A candidate with missing metadata cannot be rejected by that predicate, so it falls back to the normal heap recheck.

If you need to accelerate `IS NULL` or "missing" searches, encode missing values explicitly, for example `COALESCE(source_col::bigint, -1)`, and query that sentinel value.

## Inserts, Updates, Deletes

The resident metadata is stored in index tuples during index build and normal `INSERT` maintenance. Updates that change an indexed vector or an included metadata column create a replacement index entry with the replacement metadata. Deletes are cleaned up by normal PostgreSQL vacuum processing.

Generated stored metadata columns are the safest way to keep this correct:

```sql
ALTER TABLE items
  ADD COLUMN tenant_hash bigint
    GENERATED ALWAYS AS (hashtextextended(tenant_id, 0)) STORED;
```

If metadata values are maintained by application code or ETL jobs, missed updates can make the prefilter reject rows incorrectly. Test all insert, update, delete, and bulk-load paths before enabling the feature in production.

## Planner Behavior

The metadata predicates are ordinary SQL predicates from PostgreSQL's point of view. The planner still chooses between the vector index, scalar indexes, and sequential scan using normal costing. When `vchordrq.metadata_prefilter` is not `off`, the `vchordrq` cost estimator accounts for supported metadata predicates so the vector index is not over-priced for filtered `ORDER BY ... LIMIT` queries.

This does not force the vector index. Very selective scalar predicates can and should still choose scalar indexes when they are cheaper.

Run `ANALYZE` after creating metadata columns and after rebuilding indexes. Bad or missing table/index statistics can make PostgreSQL choose the wrong access path.

## Diagnostics

To see whether a query's metadata predicates are detected:

```sql
SET client_min_messages = log;
SET vchordrq.metadata_qual_diagnostics = on;
SET vchordrq.metadata_prefilter = reject_only;
SET vchordrq.metadata_active_columns = '';

SELECT id
FROM items
WHERE tenant_hash = hashtextextended('acme', 0)
  AND tenant_id = 'acme'
ORDER BY embedding <-> '[0,0,0]'::vector
LIMIT 10;

SET vchordrq.metadata_qual_diagnostics = off;
```

Diagnostics include the number of supported and unsupported metadata quals, detected column names, detected parameter values, and whether all scan quals are covered.

To verify correctness in staging:

```sql
SET vchordrq.metadata_prefilter_debug = on;
SET vchordrq.metadata_prefilter = reject_only;
```

If the metadata prefilter rejects a candidate that the heap predicate would have accepted, debug mode raises an error naming the metadata column.

## Troubleshooting

The query does not get faster:

- Confirm `vchordrq.prefilter = on`.
- Confirm `vchordrq.metadata_prefilter = reject_only`.
- Confirm the query contains predicates on the metadata columns, not only the original source columns.
- Confirm `vchordrq.metadata_active_columns` contains the actual declared metadata column names or is empty/all.
- Run `ANALYZE` on the table.
- Use diagnostics to confirm the metadata predicates were detected.

The query returns different rows with metadata enabled:

- Turn on `vchordrq.metadata_prefilter_debug` in staging.
- Check that metadata columns are generated or otherwise kept in sync.
- Check that lossy encodings are marked `exact = false`.
- Check that metadata predicates are conservative over-approximations of the real heap predicates.

The planner uses a scalar index instead:

- That may be correct for very selective filters.
- Compare `EXPLAIN (ANALYZE, BUFFERS)` with and without metadata prefilter.
- Check table and index `reltuples` after `ANALYZE`; zero or stale statistics can distort costs.

## Example: Feed Hash

This is a minimal end-to-end tenant/feed pattern.

```sql
CREATE TABLE candidate_search (
  id bigserial PRIMARY KEY,
  feed_id text NOT NULL,
  ftm_embedding vector(1024) NOT NULL,
  feed_id_meta_hash bigint
    GENERATED ALWAYS AS (hashtextextended(feed_id, 0)) STORED
);

CREATE INDEX candidate_search_ftm_embedding_idx
ON candidate_search
USING vchordrq (ftm_embedding vector_ip_ops)
INCLUDE (feed_id_meta_hash)
WITH (options = $$
[metadata]
columns = [
  { name = "feed_id_meta_hash", ops = ["eq", "in"], exact = false },
]
$$);

ANALYZE candidate_search;

SET vchordrq.prefilter = on;
SET vchordrq.metadata_prefilter = reject_only;
SET vchordrq.metadata_active_columns = '';

-- Literal queries can use hashtextextended('isolved-prod', 0), which PostgreSQL
-- can constant-fold. Prepared statements should bind the hash as a bigint.
SELECT id
FROM candidate_search
WHERE feed_id_meta_hash = $2::bigint
  AND feed_id = $3::text
ORDER BY ftm_embedding <#> $1::vector
LIMIT 500;
```

Bind `$2` to `hashtextextended($3, 0)` using the same hash value you would get from PostgreSQL, for example by precomputing it with `SELECT hashtextextended($1, 0)` in application setup.

The index checks `feed_id_meta_hash` while scanning vector candidates. The heap still checks `feed_id = $3`, so a rare hash collision cannot produce wrong results.
