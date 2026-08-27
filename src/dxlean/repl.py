"""Adapter for the leanprover-community REPL (JSON over stdin/stdout).

Protocol (verified against repl v4.30.0):
  request:  one JSON object, terminated by a blank line
  response: pretty-printed JSON, terminated by a blank line
  {"cmd": code, "env": id}      -> {"env": id, "sorries": [{"proofState": id, "goal": str}], "messages": [...]}
  {"tactic": t, "proofState": id} -> {"proofState": id2, "goals": [str, ...], "proofStatus": ...}
                                  |  {"message": "Lean error: ..."} on failure

The REPL must run under `lake env` from the target project directory so
LEAN_PATH resolves project modules; without it, imports silently produce an
Init-less environment where even `+` fails to parse.

REPLManager adds: a header environment, per-theorem root proof states, a
state -> proofState-id map, timeout kill/restart with replay, and full-proof
certification. Proof-state ids are process-local, so after a restart states
are reconstructed by replaying their tactic prefix from the theorem root.
"""
from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .states import LeanState, TheoremSpec, canonical_goals

BANNED_SUBSTRINGS = ("sorry", "admit", "native_decide", "stop")


class REPLError(Exception):
    pass


class REPLTimeout(REPLError):
    pass


class LeanREPL:
    """One REPL subprocess. Not thread-safe; owned by a single manager."""

    def __init__(self, project_dir: str, repl_bin: str):
        self.project_dir = project_dir
        self.repl_bin = repl_bin
        self.proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def start(self) -> None:
        self.proc = subprocess.Popen(
            ["lake", "env", self.repl_bin],
            cwd=self.project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lines = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc.stdout,), daemon=True).start()

    def _reader(self, stdout) -> None:
        for line in stdout:
            self._lines.put(line.rstrip("\n"))
        self._lines.put(None)  # EOF marker

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def request(self, obj: dict, timeout: float) -> dict:
        """Send one request, read one JSON response. Raises REPLTimeout/REPLError."""
        if not self.alive:
            raise REPLError("REPL process is not running")
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n\n")
        self.proc.stdin.flush()

        buf: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise REPLTimeout(f"REPL request timed out after {timeout}s: {obj}")
            try:
                line = self._lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if line is None:
                raise REPLError("REPL process closed stdout")
            if line == "":
                if buf:
                    return json.loads("\n".join(buf))
                continue  # skip leading blank lines
            buf.append(line)


@dataclass
class ApplyResult:
    status: str  # ok | solved | error | no_progress | timeout
    state: Optional[LeanState]
    message: str = ""


def _error_messages(resp: dict) -> list[str]:
    return [m.get("data", "") for m in resp.get("messages", []) if m.get("severity") == "error"]


class REPLManager:
    """High-level interface used by the domain: init theorems, apply tactics,
    certify finished proofs. Handles restart + replay transparently."""

    def __init__(self, project_dir: str, repl_bin: str, header: str = "",
                 tactic_timeout: float = 20.0, cmd_timeout: float = 120.0):
        self.repl = LeanREPL(project_dir, repl_bin)
        self.header = header.strip()
        self.tactic_timeout = tactic_timeout
        self.cmd_timeout = cmd_timeout
        self.header_env: Optional[int] = None
        self._root_ps: Dict[str, int] = {}                    # theorem name -> proofState id
        self._ps_ids: Dict[Tuple[str, str], int] = {}          # state key -> proofState id
        self._theorems: Dict[str, TheoremSpec] = {}
        self.n_requests = 0
        self.n_restarts = 0

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.repl.start()
        self.header_env = None
        self._root_ps.clear()
        self._ps_ids.clear()
        if self.header:
            resp = self._request({"cmd": self.header}, self.cmd_timeout)
            errs = _error_messages(resp)
            if errs or "env" not in resp:
                raise REPLError(f"header failed: {errs or resp}")
            self.header_env = resp["env"]

    def stop(self) -> None:
        self.repl.stop()

    def _restart(self) -> None:
        self.n_restarts += 1
        self.repl.stop()
        self.start()

    def _request(self, obj: dict, timeout: float) -> dict:
        self.n_requests += 1
        return self.repl.request(obj, timeout)

    # -- theorems and states -------------------------------------------------

    def init_theorem(self, thm: TheoremSpec) -> LeanState:
        """Elaborate `<statement> := by sorry` and return the root proof state."""
        self._theorems[thm.name] = thm
        ps_id, goal = self._elab_root(thm)
        state = LeanState(thm.name, (goal,), ())
        self._ps_ids[state.key] = ps_id
        return state

    def _elab_root(self, thm: TheoremSpec) -> Tuple[int, str]:
        cmd: dict = {"cmd": f"{thm.statement} := by sorry"}
        if self.header_env is not None:
            cmd["env"] = self.header_env
        resp = self._request(cmd, self.cmd_timeout)
        errs = _error_messages(resp)
        if errs:
            raise REPLError(f"theorem {thm.name} failed to elaborate: {errs}")
        sorries = resp.get("sorries", [])
        if len(sorries) != 1:
            raise REPLError(f"theorem {thm.name}: expected 1 sorry, got {len(sorries)}")
        self._root_ps[thm.name] = sorries[0]["proofState"]
        return sorries[0]["proofState"], sorries[0]["goal"]

    def _ensure_ps(self, state: LeanState) -> int:
        """Return a live proofState id for `state`, replaying its tactic prefix
        from the theorem root if the id was lost to a restart."""
        ps = self._ps_ids.get(state.key)
        if ps is not None:
            return ps
        thm = self._theorems[state.thm_name]
        if state.thm_name not in self._root_ps:
            self._elab_root(thm)
        ps: int = self._root_ps[state.thm_name]
        for tac in state.tactics:
            resp = self._request({"tactic": tac, "proofState": ps}, self.tactic_timeout)
            if "proofState" not in resp:
                raise REPLError(f"replay of {state.thm_name} failed at {tac!r}: {resp}")
            ps = resp["proofState"]
        self._ps_ids[state.key] = ps
        return ps

    # -- tactic application --------------------------------------------------

    def apply_tactic(self, state: LeanState, tactic: str) -> ApplyResult:
        tactic = tactic.strip()
        if not tactic:
            return ApplyResult("error", None, "empty tactic")
        low = tactic.lower()
        if any(b in low for b in BANNED_SUBSTRINGS):
            return ApplyResult("error", None, f"banned tactic: {tactic!r}")
        # The theorem is declared `:= by sorry`, so its own (sorry-backed) constant
        # exists in the search-time environment: referencing it is a circular proof
        # that certification would reject. Refuse it up front.
        if re.search(rf"(?<![A-Za-z0-9_.']){re.escape(state.thm_name)}(?![A-Za-z0-9_'])", tactic):
            return ApplyResult("error", None, f"self-reference to {state.thm_name!r}")
        try:
            ps = self._ensure_ps(state)
            resp = self._request({"tactic": tactic, "proofState": ps}, self.tactic_timeout)
        except REPLTimeout:
            self._restart()
            return ApplyResult("timeout", None, f"timeout: {tactic!r}")
        except REPLError as e:
            if not self.repl.alive:
                self._restart()
            return ApplyResult("error", None, str(e))

        if "proofState" not in resp:
            return ApplyResult("error", None, str(resp.get("message", resp)))
        errs = _error_messages(resp)
        if errs:
            return ApplyResult("error", None, "; ".join(errs))

        new_goals = tuple(resp.get("goals", []))
        new_state = LeanState(state.thm_name, new_goals, state.tactics + (tactic,))
        if not new_state.solved and canonical_goals(new_goals) == canonical_goals(state.goals):
            return ApplyResult("no_progress", None, tactic)
        self._ps_ids.setdefault(new_state.key, resp["proofState"])
        return ApplyResult("solved" if new_state.solved else "ok", new_state)

    # -- certification -------------------------------------------------------

    def check_full_proof(self, thm: TheoremSpec, tactics: Tuple[str, ...]) -> Tuple[bool, str]:
        """Re-elaborate the assembled proof from scratch: no errors, no sorries."""
        body = "\n".join(f"  {t}" for t in tactics)
        code = f"{thm.statement} := by\n{body}"
        cmd: dict = {"cmd": code}
        if self.header_env is not None:
            cmd["env"] = self.header_env
        try:
            resp = self._request(cmd, self.cmd_timeout)
        except REPLError as e:
            if not self.repl.alive:
                self._restart()
            return False, str(e)
        errs = _error_messages(resp)
        if errs:
            return False, "; ".join(errs)
        if resp.get("sorries"):
            return False, "proof contains sorries"
        return True, code
