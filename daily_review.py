"""Personal daily review reminders. Windows Task Scheduler can also invoke this module directly."""

import argparse
import base64
from datetime import datetime, time, timedelta
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from planner_store import PlannerStore, _get_setting, _save_setting


DEFAULT = {"enabled": True, "time": "21:30", "desktop": False}


def preferences(store):
    return {**DEFAULT, **store.get_setting("daily_review", {})}


def _due(db, now):
    config = {**DEFAULT, **_get_setting(db, "daily_review", {})}
    if not config["enabled"]:
        return None
    local = datetime.fromtimestamp(now)
    yesterday = local.date() - timedelta(days=1)
    at = time.fromisoformat(config["time"])
    first_occurrence = datetime.combine(local.date(), at).replace(fold=0).timestamp()
    # A skipped review time becomes due at the spring clock jump. Once reached in
    # the first autumn occurrence it stays due even while the clock repeats.
    reached = local.time() >= at or now >= first_occurrence
    last_due = local.date() if reached else yesterday
    # On waking or opening the planner late, offer the most recent due plan once.
    # Older missed dates stay available through the regular day navigation.
    row = db.execute("""SELECT day, COUNT(*) AS total, SUM(status='completed') AS done
                        FROM tasks WHERE archived=0 AND day BETWEEN ? AND ? AND day>?
                        GROUP BY day ORDER BY day DESC LIMIT 1""",
                     (yesterday.isoformat(), last_due.isoformat(), _get_setting(db, "daily_review_ack", ""))).fetchone()
    return dict(row) if row else None


def due(store):
    with store._db() as db:
        return _due(db, store.clock())


def acknowledge(store, day):
    with store._db(write=True) as db:
        if day > _get_setting(db, "daily_review_ack", ""):
            _save_setting(db, "daily_review_ack", day)


def deliver(store, notify, *, desktop):
    """Claim a local calendar day once across app sessions and the scheduled process."""
    token = uuid4().hex
    with store._db(write=True) as db:
        config = _get_setting(db, "daily_review", DEFAULT)
        now = store.clock()
        pending = _due(db, now)
        if not pending or pending["day"] <= _get_setting(db, "daily_review_sent", ""):
            return False
        failed = _get_setting(db, "daily_review_desktop_error", {})
        if desktop and not config.get("desktop"):
            return False
        if not desktop and config.get("desktop") and failed.get("day") != pending["day"]:
            return False
        claim = _get_setting(db, "daily_review_claim", {})
        if claim.get("expires", 0) > now:
            return False
        _save_setting(db, "daily_review_claim", {"token": token, "expires": now + 120})
    try:
        notify(pending)
    except Exception as error:
        with store._db(write=True) as db:
            if _get_setting(db, "daily_review_claim", {}).get("token") == token:
                _save_setting(db, "daily_review_claim", {})
                if desktop:
                    _save_setting(db, "daily_review_desktop_error", {"day": pending["day"], "message": str(error)})
        raise
    with store._db(write=True) as db:
        if _get_setting(db, "daily_review_claim", {}).get("token") == token:
            _save_setting(db, "daily_review_sent", pending["day"])
            _save_setting(db, "daily_review_claim", {})
            if desktop:
                _save_setting(db, "daily_review_desktop_error", {})
    return True


def reminder_text(plan):
    return f"Review your plan for {plan['day']}. {plan['done']} of {plan['total']} tasks completed."


# Pass values as environment data. A workspace path containing quotes, spaces or
# PowerShell metacharacters is never inserted into a script.
_REGISTER = r"""
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$action = New-ScheduledTaskAction -Execute $env:PLAN_REVIEW_EXE -Argument $env:PLAN_REVIEW_ARGS
$at = [datetime]::Today.Add([TimeSpan]::ParseExact($env:PLAN_REVIEW_TIME, 'hh\:mm', [CultureInfo]::InvariantCulture))
$triggers = @((New-ScheduledTaskTrigger -Daily -At $at), (New-ScheduledTaskTrigger -AtLogOn -User $user))
$options = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $env:PLAN_REVIEW_NAME -Action $action -Trigger $triggers -Principal $principal -Settings $options -Force | Out-Null
"""
_UNREGISTER = r"""
if (Get-ScheduledTask -TaskName $env:PLAN_REVIEW_NAME -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $env:PLAN_REVIEW_NAME -Confirm:$false
}
"""
_NOTIFY = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$icon = New-Object System.Windows.Forms.NotifyIcon
try {
    $icon.Icon = [System.Drawing.SystemIcons]::Information
    $icon.Text = 'Plan Smarter'
    $icon.Visible = $true
    $icon.ShowBalloonTip(7000, 'Plan Smarter', $env:PLAN_REVIEW_TEXT, [System.Windows.Forms.ToolTipIcon]::Info)
    Start-Sleep -Seconds 8
} finally { $icon.Dispose() }
"""


def _powershell(script, **settings):
    if sys.platform != "win32":
        raise RuntimeError("Windows desktop reminders require running Plan Smarter on Windows.")
    source = ("$ErrorActionPreference = 'Stop'\ntry {\n" + script +
              "\n} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }")
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                env={**os.environ, **settings}, capture_output=True, timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Windows PowerShell could not complete the reminder request.") from error
    if result.returncode:
        detail = " ".join(result.stderr.decode(errors="replace").split())[:240]
        raise RuntimeError("Windows could not complete the reminder request. " +
                           (detail or "Check your account’s Task Scheduler and notification permissions."))


def _schedule_windows(store, clock_time, enabled):
    directory = store.data_dir.resolve()
    name = "Plan Smarter daily review " + hashlib.sha256(os.path.normcase(str(directory)).encode()).hexdigest()[:16]
    if not enabled:
        _powershell(_UNREGISTER, PLAN_REVIEW_NAME=name)
        return
    interpreter = Path(sys.executable).with_name("pythonw.exe")
    args = subprocess.list2cmdline([str(Path(__file__).resolve()), "--notify", str(directory)])
    _powershell(_REGISTER, PLAN_REVIEW_NAME=name, PLAN_REVIEW_TIME=clock_time,
                PLAN_REVIEW_EXE=str(interpreter if interpreter.exists() else sys.executable),
                PLAN_REVIEW_ARGS=args)


def configure(store, clock_time, enabled, desktop):
    """Keep the in-app preference usable if local policy denies Windows registration."""
    at = time.fromisoformat(clock_time)
    clock_time = at.strftime("%H:%M")
    previous = preferences(store)
    desktop = bool(enabled and desktop)
    warning = None
    if desktop or previous["desktop"]:
        try:
            _schedule_windows(store, clock_time, desktop)
        except RuntimeError as error:
            warning = str(error)
            desktop = False
    store.set_setting("daily_review", {"time": clock_time, "enabled": enabled, "desktop": desktop})
    return warning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notify", type=Path, required=True, help="Personal workspace with planner.sqlite3")
    directory = parser.parse_args().notify
    if (directory / "planner.sqlite3").is_file():
        store = PlannerStore(directory)
        deliver(store, lambda plan: _powershell(_NOTIFY, PLAN_REVIEW_TEXT=reminder_text(plan)), desktop=True)


if __name__ == "__main__":
    main()
