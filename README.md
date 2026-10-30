# Hermes: an AI video editor for your own footage

Most business owners already film their work: a class, a lesson, an event. Almost none of them have time to edit it. Hermes does the editing.

You upload the raw clips and, if you want, describe the edit in your own words: "keep it under a minute, open on the biggest catch, upbeat pace, put our logo at the end." Hermes turns that description into editing decisions and cuts the video the way you asked. If you give no instructions, it uses sensible defaults. Want a change? Say it the same way ("slow down the middle, drop the second clip") and it re-cuts.

Hermes returns:

- **A finished video:** the best takes in order, dead air and false starts trimmed, audio levelled to YouTube loudness, and colour matched across clips.
- **Short vertical clips:** the strongest moments, cut and reframed for Instagram Reels, TikTok and YouTube Shorts.
- **Captions:** burned into the video and also saved as a separate file.

**Who it is for:** gyms, studios, coaches and anyone who films what they do and wants it posted the same week.

**How it works:** it watches the footage frame by frame to find the action, transcribes speech with Whisper, and cuts, levels and renders with ffmpeg. Claude decides which takes to keep and where the short clips start and end.

**What it will never do:** upload, post, or sign in to any account. You review every file before anything goes out.

**Tested on:** five vertical phone takes of me talking to camera (720x1280, 46 seconds in total), with the brief "Keep it under 40 seconds and upbeat. Open on the intro where I say who I am, and end on the line about sitting down to record. End card that says 'Caleb Schalesky' with 'AI systems for small business'. Two shorts." Claude, through the local Claude Code CLI, wrote the edit decision list. It kept four takes in story order, ended the intro before a false start at 9.8 seconds, and dropped one take to fit the length. Out came a 33.3-second video at -14.0 LUFS (true peak -1.5 dBTP) with burned captions and an .srt, plus two 1080x1920 shorts (8.5 s and 5.4 s), in 2 minutes 54 seconds on a laptop with Whisper on the CPU. The change request "slow down the middle, drop the second clip" re-cut it to 25.1 seconds in 1 minute 39 seconds. The middle take had no pause longer than the tight cap, so "slow down" changed its pace setting but not its length. Stills and both edit decision lists are in [`docs/`](docs/).

---

## Run it

Needs Python 3.11+, `ffmpeg`/`ffprobe` on the PATH, and `pip install -r requirements.txt`.

```bash
python -m hermes_editor path/to/clips --brief "keep it under a minute, open on the biggest moment, upbeat"
python -m hermes_editor path/to/clips --change "slow down the middle, drop the second clip"
```

Each run writes a new version folder (`hermes-out/v1`, `v2`, ...) with:

| File | What it is |
|---|---|
| `edl.json` | the edit decision list the run used |
| `video.mp4`, `video.srt` | the finished video, captions burned in, plus the caption file |
| `short-1.mp4` ... | 9:16 vertical shorts (1080x1920), captioned, each with its `.srt` |
| `report.json` | what was measured on the delivered files: length, loudness, true peak, timings |

Source clips are only ever read.

Other options: `--logo logo.png` (an image in the clips folder, put on the end card), `--shorts 0-3`, `--names "Proper Nouns"` to help the transcriber spell names, `--plan-only` to write the EDL without rendering, `--edl my-edit.json` to render a hand-edited EDL.

## The edit decision list

The headline step is turning plain words into explicit decisions. The brief becomes a JSON document that says exactly what will happen: clip order, in and out points, target length, pace, opening shot, end card and shorts. You can read it, diff it, edit it by hand and render that. A change request takes the previous EDL and rewrites it. A real one, from the test run below, is in [`docs/edl-v1.json`](docs/edl-v1.json), and the version after a change request is in [`docs/edl-v2.json`](docs/edl-v2.json).

Who writes the EDL (`--brain`, default `auto`):

1. **api**: the Anthropic API, when `ANTHROPIC_API_KEY` is set.
2. **cli**: the local Claude Code CLI in headless mode (`claude -p`), so it runs on a Claude subscription with no API key. It runs with no tools, from an empty folder.
3. **rules**: a deterministic parser for a fixed vocabulary (length, pace, opening, drop/move a clip, end card, number of shorts). It needs no model, and it is the fallback when a model reply fails validation twice.

Whatever a model returns is validated against the real footage (every clip exists, every in/out point is inside it, the opening is the first cut) and snapped so no cut lands mid-word.

## What happens to the footage

1. **Look.** For each clip: silences (`silencedetect`, -32 dB, 0.35 s), a loudness curve (`ebur128`), a motion curve from every decoded frame, exposure and colour from three regions (whole frame, top strip, left strip; the median, so a framing change cannot fake an exposure change), and word timestamps from Whisper (`small.en`). Takes that repeat another clip are marked as retakes, and a take that restarts a phrase and trails off gets an out point before the restart.
2. **Decide.** The brief and that inventory go to the brain, which returns the EDL.
3. **Render.** One ffmpeg filter graph and one encode per output. Pauses longer than the pace allows are shortened (tight 0.25 s, normal 0.5 s, relaxed 1.0 s). Exposure is matched by gamma, never an additive lift, 80% of the way to the clips' median and capped. Audio gets a high-pass, a presence EQ, a compressor and a limiter, then two-pass `loudnorm` to -14 LUFS (pass 1 measures the exact signal pass 2 receives). Captions are transcribed from the rendered video's own audio, so they cannot drift across joins. No caption straddles a join, and the burn-in uses an ASS file declared at the real frame size.
4. **Shorts.** The strongest moments, cropped to 9:16 and captioned above the bottom area that the Reels, TikTok and Shorts buttons cover.

## Tests

```bash
pip install pytest && pytest -q
```

The suite runs offline with a fake model client: EDL parsing and validation, brief to EDL, change requests, the render plan (pause caps, joins, grade, two-pass loudness, captions), and two guards. One fails the build if any upload, posting, sign-in or network-publishing code appears in the package. The other fails it if a personal path, email address or session id is committed.

## License

MIT
