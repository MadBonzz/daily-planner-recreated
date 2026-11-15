# Planning, reminders and logged work across clock changes

Six of fifteen checks passed at commit `78f8034`. All fifteen pass after the fixes. Each check uses New York civil time, the IANA zone for Ithaca. We ran the same assertions against both versions. These are deterministic calendar and database checks shared by both forecast models.

| Scenario | Earlier behavior | Verified behavior |
|---|---|---|
| Fall back, plan at 01:40 EDT until 01:30 | Returned zero available minutes | The next 01:30 is in EST, 50 real minutes away. Asking at 01:20 EST leaves 10 minutes. Both the dashboard clock and explicit IANA clock agree. |
| Spring forward, request a 02:30 cutoff | Accepted a time that never occurs | Asks for another cutoff. A scheduled 02:30 review can still arrive when the local clock reaches 03:00. |
| At 17:30, say “by 18:00”, then give a 90-minute budget | Allowed all 90 minutes | Retains the cutoff and allows 30 minutes. |
| At 17:30, give a 30-minute budget, then at 17:40 say “by 20:00” | Allowed 140 minutes | Retains the budget expiry and allows 20 minutes. |
| Ask repeatedly during a ten-minute budget | Rounded 509 remaining seconds to 9 schedulable minutes | Keeps the expiry instant across turns and allows 8 whole minutes. |
| Restart after work spanning midnight and both 01:30s | Both activity rows had the same local timestamp and read back as the same instant | Stores the offsets in the analytics view and sorts logs by their actual instants. The 01:30 entries remain an hour apart. |
| Sleep through the first 01:30 reminder and resume at 01:05 EST | Offered October 31’s review | Offers November 1’s review. Delivery and acknowledgement survive restart and the second 01:30. |

The controls cover 23-hour and 25-hour planning days, the remaining spring day, expiry of yesterday’s chat limit, one reminder delivery when the app stays open across the repeated hour, and a spring-gap reminder. A task started at 23:55 EDT and completed at 01:35 EST retains 160 minutes in a reopened SQLite store. The intervals used for the analytics test credit 300 seconds to October 31 and 720 seconds to November 1.

Run `python -m pytest -q tests/test_time_boundaries.py`. The neighboring daily review, chat and analytics suites brought the total to 66 passing tests. The separate Windows Task Scheduler integration requires an opt-in Windows host and was skipped on Linux. The reminder checks here exercise the persisted delivery and deduplication path with a callback in place of an OS notification.

[Per-scenario before/after results](checks.json)
