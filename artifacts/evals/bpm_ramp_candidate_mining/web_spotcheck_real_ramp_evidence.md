# Web Spot-Check: Real BPM Ramp Evidence

Scope: manual/subagent spot-check for high-ranking candidates from
`real_ramp_beatmapset_audit_unique_bpm_gt5.parquet`.

Verdict semantics:
- `confirmed`: public song/game metadata or chart references explicitly support a large BPM range or gradual tempo ramp.
- `likely`: evidence supports tempo variation in the mapped asset, but the public source is weaker or tied to a beatmap/edit rather than the base song.
- `not_confirmed`: local `.osu` redlines look ramp-like, but public song metadata found so far does not support a real song BPM ramp.

| Local rank | Beatmapset | Candidate | Local ramp shape | Verdict | Evidence |
|---:|---:|---|---|---|---|
| 1 | 435387 | Various Artists - Wh1teh Pack #1 / DM Ashura - memeMAX | 100 -> 573 over 114s | likely | Same redline pattern as `deltaMAX`; no independent public source found for `memeMAX` itself. |
| 2 | 2045174 | DM Ashura - deltaMAX | 100 -> 571.995 over 114s | confirmed | DDR Wiki documents `DeltaMAX` as 100-600 BPM and describes a gradual 100 -> 573 rise by 1 BPM steps: https://dancedancerevolution.fandom.com/wiki/DeltaMAX |
| 3 | 1996989 | Yuyoyuppe feat. Natsuki Karin - SICK -Yanderu EP- | 115 -> 715 over 22s | likely | osu forum listing gives the full EP map as 69-852 BPM; evidence is strongest for the mapped EP/timing, not necessarily each source track: https://osu.ppy.sh/community/forums/topics/1805771 |
| 5 | 2114541 | JVKE - this is what space feels like (Cut Ver.) | 100 -> 598 over 7s | not_confirmed | Public song metadata found by subagent points to 73 BPM / 146 double-time, and no real ramp source was found. |
| 6 | 2283870 | Designant - Designant. | 90 -> 400 over 48s | confirmed | Arcaea Wiki lists BPM 200 (75-400) and an explicit tempo sequence including 90 -> 400: https://arcaea.fandom.com/wiki/Designant. |
| 7 | 1800224 | LeaF - Aleph-0 (extended ver.) | 400 -> 201 over 44s | likely | Base `Aleph-0` has public 35-400 BPM metadata; extended-version exact ramp source was not found: https://orzmic.fandom.com/wiki/Aleph-0 |
| 8 | 286262 | Camellia - Lunatic Rough Party!! (Long Ver.) | 130 -> 400 over 15s | confirmed | SOUND VOLTEX references list 120-400 BPM and a gradual acceleration/deceleration progression: https://w.atwiki.jp/sdvx/pages/2086.html |
| 9 | 527220 | DJ YOSHITAKA - JOMANDA | 90 -> 300 over 12s | confirmed | Pop'n Music Wiki lists 195? (90-300); Moegirl describes 195 -> 90 -> 300 gradual tempo change: https://popnmusic.fandom.com/wiki/JOMANDA and https://zh.moegirl.org.cn/zh-hk/JOMANDA |
| 15 | 971561 | antiPLUR - Runengon | 178 -> 320 over 14s | confirmed | Public osu/gameplay metadata lists 174-320, matching local red timing: https://osu.ppy.sh/beatmapsets/971561 and https://www.youtube.com/watch?v=6Z67arqXosg |

Main caveat: this audit detects redline timing ramps in the dataset. That is the right signal for training/evaluating the current timing module, but it can include mapper-created timing ramps or edited audio, not only ramps documented for the original commercial song.
