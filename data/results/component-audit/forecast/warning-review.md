# Manual warning review

For each paired case, inspect the proposed action and its cited observation alongside the current deliverable. A source ID supports attribution. A useful action must also address an applicable obstacle. The review records unsupported procedures or explanations even when the topic is correct. These are four selected synthetic cases, including cases used to develop the warning prompt.

| Model and task | First revision | Warnings-only trial |
| --- | --- | --- |
| 4B, Mara handover | Copied an unrelated hardware-pairing check. The warning also mentioned the documented late arrival. | Confirming Mara's expected arrival addresses the note. The claim that coordination delays caused her lateness is unverified. |
| 4B, dishwashing | Copied the hardware-pairing check in response to a skillet needing a second scrub. | Prescribed drying cast iron **before** scrubbing. The user mentioned drying a pan and the prior note mentioned a second scrub, but neither supports this sequence. |
| 4B, Dev stock check | Copied the hardware-pairing check from a note about a prepared spreadsheet. | Suggested skipping the initial review. Dev's preparation supports a shorter forecast. The note gives no obstacle to warn about, and agreeing quantities still requires reviewing the sheet. |
| 4B, camera dark frames | Recommended checking the controller's USB pairing. The prior note directly records two pairing failures. | Recommended re-pairing the cooling controller. This still addresses the documented obstacle. |
| 9B, Mara handover | Copied the hardware-pairing instruction from the arrival note. | Offered a pertinent arrival-time check. A second warning claimed that **Mara** played a crossword, citing ActivityWatch on the planner user's device. The log does not identify Mara as the actor. |
| 9B, dishwashing | Copied the hardware-pairing instruction. | Warned about a phone game played during the second scrub. The source records three minutes of Candy Crush. The separate note records a second scrub. Neither establishes that they occurred at the same time. |
| 9B, Dev stock check | Copied the hardware-pairing instruction. | Suggested pre-filtering the sheet, directly related to the previous early finish. This may help explain a shorter forecast. The note gives no observed obstacle. |
| 9B, camera dark frames | Repeated the demonstration wording about checking pairing. In this case the warning is supported by the cited USB controller failures. | The controller pairing and stabilization checks are grounded in two notes describing repeated pairing failures. Both warnings address the same issue. |

On 4B the copied demonstration appeared without pairing evidence in three of four first-revision cases and in zero of four warning-trial cases. The alternative gave a pertinent arrival check on Mara's meeting, while leaving a speculative causal claim in the same warning. It still supplied unsupported advice on dishes and the stock check. The warning instruction also changed Mara's timing from 40 to 35 minutes, moving it outside the predeclared 1.23–1.44× band for a 30-minute budget.

The same three inappropriate pairing instructions disappeared on the 9B trial. The 9B trial correctly referred to the game and crossword observations yet attributed a time overlap and person that ActivityWatch could not establish.

[First-revision 4B receipts](revised/bonsai-4b/) · [Warning-trial 4B receipts](warning-trial/bonsai-4b/) · [First-revision 9B receipts](revised/bonsai-9b/) · [Warning-trial 9B receipts](warning-trial/bonsai-9b/)

## Combined instruction replay

The confidence sentence can also change warning generation. This separate replay used all four warning-panel tasks with the combined forecast instruction.

| Model and task | Combined instruction |
| --- | --- |
| 4B, Mara handover | Advises confirming the start and budgeting for late arrival. The eight-minute delay is documented. Its proposed cause, coordination delays, is unverified. |
| 4B, dishwashing | Recognizes the skillet needed a second scrub. Pre-scraping with a stiff brush is a new suggestion. The source does not establish it shortened this user's cleaning time. |
| 4B, Dev stock check | Cites the filtered stock sheet and again recommends skipping the initial review. The note records preparation and an early finish. It gives no reason the reorder review can be omitted. |
| 4B, camera dark frames | Recommends checking the cooling controller’s USB pairing before capture, addressing the documented pairing failures. |
| 9B, Mara handover | Advises aligning the start with Mara's expected arrival, supported by the late-arrival note. A second warning advises removing the phone. The crossword observation identifies a site and window title, with no phone identified. |
| 9B, dishwashing | Accurately cites three and five minutes of phone gaming. It asserts those exact windows coincided with scrubbing a skillet and handles. The app recorded no timestamps for those kitchen actions. |
| 9B, Dev stock check | Recommends pre-filtering the order sheet. This reflects a previously successful preparation step. The source gives no observed delay from re-sorting. |
| 9B, camera dark frames | Recommends checking controller pairing and a stable connection before capture. Both cited notes report loss of pairing. The two warnings address the same issue. |

Across the four repeated tasks per model, the first revision copied the unrelated demonstration on six of eight forecasts. The warnings-only and combined instructions each produced zero copies over the same eight forecasts. Both models retained the genuine camera warning in all three versions. In the combined run both still issued a stock-check warning based on a successful preparation step, and the dishwashing and meeting explanations contained unsupported details. These checks establish the specific change in wording and its limits on the repeat panel.

[Combined 4B receipts](confidence-trial-warning-panel/bonsai-4b/) · [Combined 9B receipts](confidence-trial-warning-panel/bonsai-9b/)
