# Five editable demo histories

Choose **Dev** and select a person. Open **History details** to visit a past day or add one of the three prospective tasks. Each person has six unfinished tasks, including work already recorded. Mae and Leila have a running timer when their schedules open.

| Person | Completed | Notes on completion | Calendar actions | Native watcher events | What to look for |
| --- | ---: | ---: | ---: | ---: | --- |
| Nora, distributed systems student | 16 | 10 | 84 | 444 | Compare her C-to-Rust debugging sessions with writing up a saved trace. She pauses a long mismatch investigation and resumes after lab. |
| Ishan, GPU researcher | 16 | 8 | 86 | 471 | Reproducing a failing kernel can span two GPU bookings. A figures-only reading has a different pattern from a board derivation. |
| Mae, clinic and playhouse volunteer | 19 | 19 | 85 | 262 | Compare calls with Mara and Dev. Follow intake coverage through to that evening’s caption shift. A scheduled performance ends at the final cue. |
| Leila, signals student and audio volunteer | 19 | 15 | 95 | 439 | Open a Fourier lecture to see course videos and unrelated recommendations on the same domain. Her lunch-hour problem solving and fixed station shifts provide different comparisons. |
| Xavier, robotics club member and housemate | 19 | 19 | 98 | 295 | Compare a live rover capture with a saved bag replay. While a wash cycle is paused, he can work on the replay. Long physical stretches have little observable device activity. |

## Dates and recorded time

The authored calendars end on Saturday, March 21, 2026 in New York local time. When loaded, day zero moves to the local day you open the history. Earlier tasks retain their day offsets and local clock times. Some tasks were created several days before they were started. The timer history includes explicit pauses, gaps between sessions, and later completions. The source timestamp of each session carries its UTC offset, including sessions on opposite sides of the spring clock change.

The 448 actions cover 89 completions and 30 open tasks. Fifteen further examples can be added from the task picker. Across the five profiles, 34 task records refer to existing PDFs, including lab handouts and papers. These reuse files from the [PDF collection](../pdfs). Each history mixes previously benchmarked task descriptions with two newly written completed tasks and one new open task. [Source mappings](sources.json) identify the benchmark text used for each reused task.

## What appears in a report

The [ActivityWatch SQLite stores](activitywatch) contain 1,911 synthetic desktop window, browser tab, AFK and Android foreground events. Five separate phone hostnames identify the device that supplied each mobile event. The stores also include ordinary activity outside the timers. Each task report reads only its own timer periods and pairs them with the fixture’s reviewed annotations. Asking the app to rebuild a report exercises the configured classifier.

The 89 completed task reports expose 1,412 activity intervals, including laptop-only, phone-only and simultaneous device use in every history. The 32-minute saved-log write-up has nine intervals. Forty of the other long computer tasks have at least ten. Here a long computer task means at least 30 recorded minutes, a laptop foreground for at least half the time, and an open-ended duration. Dishwashing can have long unobserved stretches, and a recording that plays continuously can show a single foreground window.

The [histories](histories) hold pre-task inputs, notes and timer annotations. The [schedules](schedules) give the order in which the planner receives each creation and timer event. Forecasts run through the dashboard when you open an editable history.
