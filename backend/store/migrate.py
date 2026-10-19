"""Apply supabase/migrations/*.sql to a Postgres database.

For Supabase, prefer the CLI (`supabase db push`); this helper is for local
Postgres and the integration tests. It records applied versions in the same
table the Supabase CLI uses (`supabase_migrations.schema_migrations`), so a
database migrated here also looks migrated to the CLI. On a plain Postgres it
first installs stand-ins for Supabase's roles and `auth.uid()`.

    python -m backend.store.migrate            # uses DATABASE_URL
    python -m backend.store.migrate postgresql://...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "supabase" / "migrations"
SHIM = Path(__file__).parent / "sql" / "local_supabase_shim.sql"


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    return sorted(p for p in directory.glob("*.sql") if p.name.split("_", 1)[0].isdigit())


def apply_migrations(database_url: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations in version order, each in its own transaction. Returns applied versions."""
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True, prepare_threshold=None) as conn:
        has_auth_uid = conn.execute(
            "select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
            "where n.nspname = 'auth' and p.proname = 'uid'"
        ).fetchone()
        if not has_auth_uid:
            conn.execute(SHIM.read_text())
        conn.execute("create schema if not exists supabase_migrations")
        conn.execute(
            "create table if not exists supabase_migrations.schema_migrations "
            "(version text primary key, statements text[], name text)"
        )
        done = {r[0] for r in conn.execute("select version from supabase_migrations.schema_migrations")}
        for path in migration_files(directory):
            version, _, name = path.stem.partition("_")
            if version in done:
                continue
            sql = path.read_text()
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "insert into supabase_migrations.schema_migrations (version, statements, name) values (%s, %s, %s)",
                    (version, [sql], name),
                )
            applied.append(version)
    return applied


def main(argv: list[str]) -> int:
    url = argv[1] if len(argv) > 1 else os.environ.get("DATABASE_URL")
    if not url:
        print("usage: python -m backend.store.migrate [DATABASE_URL]", file=sys.stderr)
        return 2
    applied = apply_migrations(url)
    print(f"applied {len(applied)} migration(s): {', '.join(applied) or 'none pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
