# Agent Headroom brand

Suite name: **AGENT HEADROOM**. Headroom is the margin before clipping; every test measures a different kind of margin for a fleet of agents. Wordmark: AGENT in green IBM Plex Mono (letterspaced) above HEADROOM in Sora 700 caps (`agent-headroom-wordmark.svg`). One-line lockup: mark, `AGENT HEADROOM`, then the test name after a middle dot in Plex Mono (`AGENT HEADROOM · RECALL`). Each test card carries a kicker naming its headroom. CLI ids are unchanged.

| test (CLI id) | brand name | kicker | icon | the margin it measures |
|---|---|---|---|---|
| ladder | RUNG | size headroom | rung.svg (ladder, highlighted rung, block on it) | the largest model and context that fit |
| inference | PACE | speed headroom | pace.svg (pulse line in a frame) | decode and end-to-end speed on the frozen replay |
| host_pressure | CROWD | concurrency headroom | crowd.svg (agents against a wall, last one red) | concurrent agents before the first failure |
| residency | TENANTS | model headroom | tenants.svg (four rooms, one evicted) | models kept loaded, swap cost |
| warm_contexts | RECALL | context headroom | recall.svg (parked contexts, one restored to a slot) | agent contexts kept warm in host RAM |
| coherent topology | BRIDGE | badge | bridge.svg | HBM plus LPDDR over NVLink-C2C |
| clusters | FLEET | badge | fleet.svg | several machines, one benchmark |
| engine rows | TRACKS | badge | tracks.svg | llama.cpp baseline plus the fastest engine |

Icons: 24-unit grid, 1.6 stroke on `currentColor`, one accent each. Colors (dark theme): GPU #39C6F2, host RAM and headroom #F6A84B, KV and context #B58CFF, weights #3F86FF, agents and ok #4FD98A, evicted and failed #FF5E6C, panel #111820 on #0A0E13, text #E8EEF4, muted #93A1B0. Light theme uses the same roles at lower chroma (see docs/explainer.html). Brand sheet, icons at scale and a 1920×1080 title card: the "Agent Headroom Brand" design canvas.
