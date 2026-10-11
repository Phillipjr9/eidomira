"""Grant or revoke administrator access on this installation.

    python -m tools.grant_admin --list
    python -m tools.grant_admin you@example.com
    python -m tools.grant_admin --demote you@example.com

The first administrator is made here rather than by registering with a special password,
because a documented default account is a back door that survives every later review: it
gets forgotten, it does not get changed, and it is in the repository. So the flow is
ordinary registration through the site, then this command run on the server by whoever
already has shell access — which is the same person who could read the database anyway.

Runs against the database the application is configured with, and says which one that is,
because promoting an account on the wrong database looks exactly like a command that
worked.
"""
from __future__ import annotations

import argparse
import sys

from app.config import settings
from app.database import database


def _rows(sql: str, values=()) -> list[dict]:
    """Read inside an open connection.

    `database.execute()` closes its connection before returning the cursor, so fetching
    from the result raises "Cannot operate on a closed database" — the failure `--list`
    had the first time it ran.
    """
    with database.lock, database.connect() as db:
        return [dict(row) for row in db.execute(sql, values).fetchall()]


def _find(email: str) -> dict | None:
    return database.one("SELECT id,email,role,disabled FROM users WHERE email=?",
                        (email.strip().lower(),))


def _admin_count() -> int:
    return database.one("SELECT COUNT(*) AS n FROM users WHERE role='admin' AND disabled=0")["n"]


def _set_role(user: dict, role: str) -> None:
    database.execute("UPDATE users SET role=? WHERE id=?", (role, user["id"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("email", nargs="?",
                        help="account to promote; omit with --list")
    parser.add_argument("--demote", action="store_true",
                        help="take administrator access away instead of granting it")
    parser.add_argument("--list", action="store_true", help="show every account and its role")
    parser.add_argument("--database", help="override the configured database path")
    args = parser.parse_args(argv)

    if args.database:
        from pathlib import Path
        database.path = Path(args.database)

    # Always say which database, before doing anything to it.
    print(f"database: {database.path}")

    if args.list:
        rows = _rows("SELECT email,role,disabled,email_verified_at FROM users ORDER BY email")
        if not rows:
            print("no accounts yet — register through the site first")
            return 0
        width = max(len(row["email"]) for row in rows)
        for row in rows:
            flags = []
            if row["role"] == "admin":
                flags.append("administrator")
            if row["disabled"]:
                flags.append("disabled")
            if not row["email_verified_at"]:
                flags.append("unverified")
            print(f"  {row['email']:<{width}}  {' · '.join(flags) or 'user'}")
        print(f"\n{_admin_count()} administrator(s)")
        return 0

    if not args.email:
        parser.error("give an email address, or use --list")

    user = _find(args.email)
    if not user:
        # Not created for you: an account made from a shell prompt would have a password
        # that travelled through a terminal, a shell history or a chat log.
        print(f"no account for {args.email.strip().lower()} — register through the site, "
              f"then run this again", file=sys.stderr)
        return 1

    if args.demote:
        if user["role"] != "admin":
            print(f"{user['email']} is not an administrator; nothing to do")
            return 0
        if _admin_count() <= 1:
            # Refusing here is the difference between a mistake and an installation with
            # nobody who can administer it.
            print("refusing: this is the last administrator. Promote someone else first.",
                  file=sys.stderr)
            return 1
        _set_role(user, "user")
        print(f"demoted {user['email']} to an ordinary account")
        return 0

    if user["role"] == "admin":
        print(f"{user['email']} is already an administrator")
        return 0
    if user["disabled"]:
        print(f"warning: {user['email']} is disabled and cannot sign in until that changes")
    _set_role(user, "admin")
    print(f"promoted {user['email']} to administrator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
