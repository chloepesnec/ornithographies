from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

try:
    import librosa
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: librosa. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
IMAGE_EXTENSIONS = {".svg", ".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_DISPLAY_CONFIG = {
    "timeline_height": 360,
    "staff_top": 92,
    "line_spacing": 44,
    "px_per_second": 110,
    "note_height": 128,
    "note_min_width": 128,
    "active_scale": 1.25,
    "show_note_labels": False,
    "highlight_color": "rgba(224, 90, 42, 0.55)",
    "vertical_padding": 8,
    "pitch_step_px": 22,
}


def hz_to_midi(freq_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(freq_hz / 440.0)


def midi_to_note_name(midi_value: float) -> str:
    midi_int = int(round(midi_value))
    octave = midi_int // 12 - 1
    return f"{NOTE_NAMES[midi_int % 12]}{octave}"


def midi_to_frequency(midi_value: float) -> float:
    return 440.0 * (2.0 ** ((midi_value - 69.0) / 12.0))


def smooth_midi(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values

    result = values.copy()
    half = window // 2
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        chunk = values[start:end]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            result[index] = np.median(chunk)
    return result


def extract_notes(
    audio_path: Path,
    min_note_duration: float,
    max_gap: float,
    min_voiced_probability: float,
    note_change_semitones: float,
    frame_length: int,
    hop_length: int,
    fmin: str,
    fmax: str,
) -> tuple[list[dict], float]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    fmin_hz = float(librosa.note_to_hz(fmin))
    fmax_hz = float(librosa.note_to_hz(fmax))

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    midi = np.full_like(f0, np.nan, dtype=float)
    valid = np.isfinite(f0) & voiced_flag & (voiced_prob >= min_voiced_probability)
    midi[valid] = [hz_to_midi(float(freq)) for freq in f0[valid]]
    midi = smooth_midi(midi, window=5)

    notes: list[dict] = []
    current_frames: list[int] = []
    current_pitch: float | None = None
    last_time: float | None = None

    def flush() -> None:
        nonlocal current_frames, current_pitch, last_time
        if not current_frames:
            return

        start = float(times[current_frames[0]])
        end = float(times[current_frames[-1]] + hop_length / sr)
        if end - start >= min_note_duration:
            frame_midi = midi[current_frames]
            frame_prob = voiced_prob[current_frames]
            pitch = float(np.nanmedian(frame_midi))
            rounded_midi = int(round(pitch))
            notes.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "note": midi_to_note_name(rounded_midi),
                    "midi": rounded_midi,
                    "frequency_hz": round(midi_to_frequency(rounded_midi), 2),
                    "estimated_midi": round(pitch, 2),
                    "confidence": round(float(np.nanmedian(frame_prob)), 3),
                }
            )

        current_frames = []
        current_pitch = None
        last_time = None

    for index, pitch in enumerate(midi):
        if not np.isfinite(pitch):
            if last_time is not None and float(times[index]) - last_time > max_gap:
                flush()
            continue

        if not current_frames:
            current_frames = [index]
            current_pitch = float(pitch)
            last_time = float(times[index])
            continue

        assert current_pitch is not None
        gap = float(times[index]) - float(last_time)
        pitch_delta = abs(float(pitch) - current_pitch)
        if gap <= max_gap and pitch_delta <= note_change_semitones:
            current_frames.append(index)
            current_pitch = float(np.nanmedian(midi[current_frames]))
            last_time = float(times[index])
        else:
            flush()
            current_frames = [index]
            current_pitch = float(pitch)
            last_time = float(times[index])

    flush()
    if notes:
        return notes, duration

    return extract_energy_notes(
        y=y,
        sr=sr,
        min_note_duration=min_note_duration,
        max_gap=max_gap,
        hop_length=hop_length,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    ), duration


def extract_energy_notes(
    y: np.ndarray,
    sr: int,
    min_note_duration: float,
    max_gap: float,
    hop_length: int,
    fmin_hz: float,
    fmax_hz: float,
) -> list[dict]:
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    if not len(rms) or float(np.max(rms)) <= 0:
        return []

    threshold = max(float(np.percentile(rms, 75)) * 1.35, float(np.max(rms)) * 0.08)
    active = rms >= threshold

    segments: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None
    max_gap_frames = max(1, int(round(max_gap * sr / hop_length)))

    for index, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = index
            last_active = index
            continue

        if start is not None and last_active is not None and index - last_active > max_gap_frames:
            segments.append((start, last_active))
            start = None
            last_active = None

    if start is not None and last_active is not None:
        segments.append((start, last_active))

    notes: list[dict] = []
    for start_frame, end_frame in segments:
        start = float(times[start_frame])
        end = float(times[end_frame] + hop_length / sr)
        if end - start < min_note_duration:
            continue

        sample_start = max(0, int(start * sr))
        sample_end = min(len(y), int(end * sr))
        segment = y[sample_start:sample_end]
        if len(segment) < 256:
            continue

        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
        freq = float(np.nanmedian(centroid))
        freq = min(max(freq, fmin_hz), fmax_hz)
        midi_value = int(round(hz_to_midi(freq)))
        notes.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "note": midi_to_note_name(midi_value),
                "midi": midi_value,
                "frequency_hz": round(midi_to_frequency(midi_value), 2),
                "estimated_midi": round(hz_to_midi(freq), 2),
                "confidence": 0.25,
                "method": "energy_centroid",
            }
        )

    return notes


def write_csv(path: Path, notes: list[dict]) -> None:
    fields = [
        "start",
        "end",
        "duration",
        "note",
        "midi",
        "frequency_hz",
        "estimated_midi",
        "confidence",
        "method",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(notes)


def relative_url(path: Path, target: Path) -> str:
    relative = os.path.relpath(target, path.parent).replace("\\", "/")
    return html.escape(quote(relative, safe="/:"))


def collect_note_images(audio_path: Path, note_image: str | None) -> list[Path]:
    if note_image:
        selected = Path(note_image)
    else:
        selected = Path("img") / audio_path.stem
        if not selected.exists():
            selected = Path("img") / "triangle 01.svg"

    if selected.is_dir():
        images = sorted(
            path
            for path in selected.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif selected.is_file():
        images = [selected]
    else:
        images = []

    if not images:
        raise SystemExit(f"No note image found for: {selected}")
    return images


def load_display_config(config_path: Path, audio_path: Path) -> dict:
    config = DEFAULT_DISPLAY_CONFIG.copy()
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(raw.get("default", {}))
        config.update(raw.get("birds", {}).get(audio_path.stem, {}))
    return config


def write_html(
    path: Path,
    audio_path: Path,
    notes: list[dict],
    duration: float,
    note_image_paths: list[Path],
    display_config: dict,
) -> None:
    rel_audio = relative_url(path, audio_path)
    rel_note_images = [relative_url(path, image_path) for image_path in note_image_paths]
    data = json.dumps({"duration": duration, "notes": notes}, ensure_ascii=True)
    image_data = json.dumps(rel_note_images, ensure_ascii=True)
    config_data = json.dumps(display_config, ensure_ascii=True)
    page = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bird notes</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, sans-serif; }}
    body {{ margin: 0; background: #f7f7f4; color: #202020; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    audio {{ width: 100%; margin: 12px 0 22px; }}
    .timeline {{ position: relative; height: var(--timeline-height); overflow-x: auto; border: 1px solid #ccc; background: #fff; }}
    .track {{ position: relative; height: 100%; min-width: 900px; }}
    .guide {{ position: absolute; left: 0; right: 0; height: 1px; background: #deded8; }}
    .note {{
      position: absolute;
      width: var(--note-width);
      height: var(--note-height);
      padding: 0;
      border: 0;
      background: transparent;
      transform: translate(-50%, -50%);
      opacity: 0;
      cursor: pointer;
      pointer-events: none;
      transition: opacity .08s linear, transform .08s linear, filter .08s linear;
    }}
    .note.played {{
      opacity: .55;
      pointer-events: auto;
    }}
    .note img {{ width: 100%; height: 100%; display: block; pointer-events: none; }}
    .note.active {{
      opacity: 1;
      transform: translate(-50%, -50%) scale(var(--active-scale));
      filter: drop-shadow(0 0 10px var(--highlight-color));
      z-index: 2;
    }}
    .label {{ position: absolute; transform: translate(-50%, 24px); font-size: 12px; white-space: nowrap; color: #555; }}
    .label {{ opacity: 0; }}
    .label.played {{ opacity: 1; }}
    .playhead {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #111; pointer-events: none; }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(audio_path.stem)}</h1>
    <audio id="audio" controls src="{rel_audio}"></audio>
    <div class="timeline" id="timeline"><div class="track" id="track"></div></div>
  </main>
  <script>
    const DATA = {data};
    const NOTE_IMAGES = {image_data};
    const CONFIG = {config_data};
    const track = document.getElementById('track');
    const audio = document.getElementById('audio');
    const timeline = document.getElementById('timeline');
    const pxPerSecond = CONFIG.px_per_second;
    const width = Math.max(900, DATA.duration * pxPerSecond + 80);
    const timelineHeight = CONFIG.timeline_height;
    const staffTop = CONFIG.staff_top;
    const lineSpacing = CONFIG.line_spacing;
    const staffCenterMidi = 71;
    const noteHeight = CONFIG.note_height;
    const noteMinWidth = CONFIG.note_min_width;
    const activeScale = CONFIG.active_scale;
    const showNoteLabels = Boolean(CONFIG.show_note_labels);
    const highlightColor = CONFIG.highlight_color;
    const noteHalfHeight = noteHeight * activeScale / 2;
    const minY = CONFIG.vertical_padding + noteHalfHeight;
    const maxY = timelineHeight - CONFIG.vertical_padding - noteHalfHeight;
    const centerY = staffTop + 2 * lineSpacing;
    const midiValues = DATA.notes.map((item) => item.midi);
    const minMidi = midiValues.length ? Math.min(...midiValues) : staffCenterMidi;
    const maxMidi = midiValues.length ? Math.max(...midiValues) : staffCenterMidi;
    const midiCenter = (minMidi + maxMidi) / 2;
    const midiRange = Math.max(1, maxMidi - minMidi);
    const availablePitchHeight = Math.max(1, maxY - minY);
    const pitchStep = Math.min(CONFIG.pitch_step_px, availablePitchHeight / midiRange);
    timeline.style.setProperty('--timeline-height', timelineHeight + 'px');
    track.style.setProperty('--note-height', noteHeight + 'px');
    track.style.setProperty('--active-scale', activeScale);
    track.style.setProperty('--highlight-color', highlightColor);
    track.style.width = width + 'px';

    function pseudoRandom(seed) {{
      const x = Math.sin(seed * 999.91) * 10000;
      return x - Math.floor(x);
    }}

    for (let i = 0; i < 5; i++) {{
      const line = document.createElement('div');
      line.className = 'guide';
      line.style.top = (staffTop + i * lineSpacing) + 'px';
      track.appendChild(line);
    }}

    const playhead = document.createElement('div');
    playhead.className = 'playhead';
    track.appendChild(playhead);

    const elements = DATA.notes.map((item) => {{
      const x = item.start * pxPerSecond + 40;
      const rawY = centerY - ((item.midi - midiCenter) * pitchStep);
      const y = Math.min(maxY, Math.max(minY, rawY));
      const noteWidth = Math.max(noteMinWidth, item.duration * pxPerSecond);
      const note = document.createElement('button');
      note.className = 'note';
      note.type = 'button';
      note.title = `${{item.note}} ${{item.start}}s`;
      note.style.left = x + 'px';
      note.style.top = y + 'px';
      note.style.setProperty('--note-width', noteWidth + 'px');
      const image = document.createElement('img');
      image.alt = item.note;
      const imageIndex = Math.floor(pseudoRandom(item.start + item.midi + item.duration) * NOTE_IMAGES.length);
      image.src = NOTE_IMAGES[imageIndex];
      note.appendChild(image);
      note.addEventListener('click', () => {{
        audio.currentTime = item.start;
        audio.play();
      }});
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = item.note;
      label.style.left = x + 'px';
      label.style.top = y + 'px';
      label.hidden = !showNoteLabels;
      track.append(note, label);
      return {{ item, note, label }};
    }});

    function tick() {{
      const time = audio.currentTime || 0;
      playhead.style.left = (time * pxPerSecond + 40) + 'px';
      for (const entry of elements) {{
        const played = time >= entry.item.start;
        const active = time >= entry.item.start && time <= entry.item.end;
        entry.note.classList.toggle('played', played);
        if (showNoteLabels) {{
          entry.label.classList.toggle('played', played);
        }}
        entry.note.classList.toggle('active', active);
      }}
      requestAnimationFrame(tick);
    }}
    tick();
  </script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract approximate musical notes and timecodes from a bird sound."
    )
    parser.add_argument("audio", nargs="?", default="sound/canard.mp3")
    parser.add_argument("--out-dir", default="analysis")
    parser.add_argument("--min-note-duration", type=float, default=0.08)
    parser.add_argument("--max-gap", type=float, default=0.06)
    parser.add_argument("--min-voiced-probability", type=float, default=0.35)
    parser.add_argument("--note-change-semitones", type=float, default=1.25)
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--fmin", default="C3")
    parser.add_argument("--fmax", default="C7")
    parser.add_argument(
        "--note-image",
        default=None,
        help="Image file or directory used for notes. Defaults to img/<audio-name>/ when available.",
    )
    parser.add_argument("--display-config", default="display_config.json")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes, duration = extract_notes(
        audio_path=audio_path,
        min_note_duration=args.min_note_duration,
        max_gap=args.max_gap,
        min_voiced_probability=args.min_voiced_probability,
        note_change_semitones=args.note_change_semitones,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    payload = {
        "audio": audio_path.as_posix(),
        "duration": round(duration, 3),
        "notes": notes,
    }
    base = out_dir / audio_path.stem
    json_path = base.with_suffix(".notes.json")
    csv_path = base.with_suffix(".notes.csv")
    html_path = base.with_suffix(".html")

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    write_csv(csv_path, notes)
    note_images = collect_note_images(audio_path, args.note_image)
    display_config = load_display_config(Path(args.display_config), audio_path)
    write_html(html_path, audio_path, notes, duration, note_images, display_config)

    print(f"Audio: {audio_path}")
    print(f"Duration: {duration:.3f}s")
    print(f"Detected notes: {len(notes)}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"HTML player: {html_path}")


if __name__ == "__main__":
    main()
