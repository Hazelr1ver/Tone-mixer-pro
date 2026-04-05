import threading
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import ttk, messagebox

import numpy as np
import pyaudio

# Settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
WAVE_TYPES = ("sine", "square", "sawtooth", "triangle")

THEMES = {
    "dark": {
        "bg": "#0f141a",
        "panel": "#151c24",
        "card": "#19212b",
        "canvas": "#0a0f14",
        "fg": "#dbe5ee",
        "muted": "#8ea0b2",
        "edge": "#263544",
        "accent": "#65c9ff",
        "grid": "#22313d",
        "trace": ["#17303d", "#21465a", "#2f6f8f", "#63c6ff"],
    },
    "light": {
        "bg": "#eef2f6",
        "panel": "#ffffff",
        "card": "#ffffff",
        "canvas": "#f8fafc",
        "fg": "#1e2933",
        "muted": "#5e6b78",
        "edge": "#c7d1db",
        "accent": "#0078d4",
        "grid": "#d9e1e8",
        "trace": ["#b5d6ea", "#7fb8db", "#3d94c6", "#0078d4"],
    },
}

# Wave generation
def waveform_from_phase(wave_type: str, phase: np.ndarray) -> np.ndarray:
    two_pi = 2.0 * np.pi

    if wave_type == "sine":
        return np.sin(phase)
    if wave_type == "square":
        return np.where(np.sin(phase) >= 0.0, 1.0, -1.0)
    if wave_type == "sawtooth":
        x = (phase / two_pi) % 1.0
        return 2.0 * x - 1.0
    if wave_type == "triangle":
        x = (phase / two_pi) % 1.0
        return 4.0 * np.abs(x - 0.5) - 1.0

    return np.sin(phase)


def render_tone_samples(freq: float, wave_type: str, volume: float, phase0: float, n_samples: int) -> np.ndarray:
    n = np.arange(n_samples, dtype=np.float32)
    phase = phase0 + (2.0 * np.pi * freq * n / SAMPLE_RATE)
    return waveform_from_phase(wave_type, phase) * volume


def render_mix_samples(snapshot, n_samples: int) -> np.ndarray:
    mix = np.zeros(n_samples, dtype=np.float32)

    for tone_id, freq, wave_type, volume, phase0, muted in snapshot:
        if muted:
            continue
        mix += render_tone_samples(freq, wave_type, volume, phase0, n_samples)

    # fixed headroom instead of per-buffer normalization
    mix *= 0.25

    return mix


def samples_to_points(y: np.ndarray, w: int, h: int, y_scale: float = 0.40):
    mid = h // 2
    n = len(y)
    points = []
    for i, val in enumerate(y):
        px = i * (w - 1) / (n - 1)
        py = mid - val * (h * y_scale)
        points.extend([px, py])
    return points

# Shared tone state
@dataclass
class Tone:
    tone_id: int
    freq: float
    wave_type: str
    volume: float
    initial_volume: float
    phase: float = 0.0
    muted: bool = False
    row: "ToneRow | None" = field(default=None, repr=False)


tone_lock = threading.Lock()
tones: list[Tone] = []
next_tone_id = 1
stream_running = True
audio_paused = False
master_volume = 0.5

# Audio callback
def audio_callback(in_data, frame_count, time_info, status):
    global stream_running, audio_paused, master_volume

    if not stream_running:
        return (None, pyaudio.paComplete)

    if audio_paused:
        waveform = np.zeros(frame_count, dtype=np.float32)
    else:
        with tone_lock:
            snapshot = [
                (t.tone_id, t.freq, t.wave_type, t.volume, t.phase, t.muted)
                for t in tones
            ]

        if snapshot:
            waveform = render_mix_samples(snapshot, frame_count)
            waveform *= master_volume

            phase_increment_base = 2.0 * np.pi * frame_count / SAMPLE_RATE
            with tone_lock:
                tone_map = {t.tone_id: t for t in tones}
                for tone_id, freq, wave_type, volume, phase0, muted in snapshot:
                    tone = tone_map.get(tone_id)
                    if tone is not None:
                        tone.phase = (phase0 + phase_increment_base * freq) % (2.0 * np.pi)
        else:
            waveform = np.zeros(frame_count, dtype=np.float32)

    int_samples = np.clip(waveform, -1.0, 1.0)
    int_samples = (int_samples * np.iinfo(np.int16).max).astype(np.int16)
    return (int_samples.tobytes(), pyaudio.paContinue)
    
# Theme setup
def configure_theme(root, mode: str):
    palette = THEMES[mode]
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=palette["bg"], foreground=palette["fg"], fieldbackground=palette["panel"])
    style.configure("TFrame", background=palette["bg"])
    style.configure("Card.TFrame", background=palette["card"])
    style.configure("TLabel", background=palette["bg"], foreground=palette["fg"])
    style.configure("Muted.TLabel", background=palette["bg"], foreground=palette["muted"])
    style.configure("TLabelframe", background=palette["bg"], foreground=palette["fg"])
    style.configure("TLabelframe.Label", background=palette["bg"], foreground=palette["fg"])
    style.configure("TButton", background=palette["panel"], foreground=palette["fg"], borderwidth=1)
    style.map("TButton", background=[("active", palette["edge"]), ("pressed", palette["edge"])])
    style.configure("TEntry", fieldbackground=palette["panel"], foreground=palette["fg"], insertcolor=palette["fg"])
    style.configure("TCombobox", fieldbackground=palette["panel"], foreground=palette["fg"])
    style.map("TCombobox", fieldbackground=[("readonly", palette["panel"])], foreground=[("readonly", palette["fg"])])
    style.configure("TCheckbutton", background=palette["bg"], foreground=palette["fg"])
    style.configure("Horizontal.TScale", background=palette["bg"])

    root.configure(bg=palette["bg"])
    return palette

# Tone row
class ToneRow:
    def __init__(self, parent, app, tone: Tone):
        self.app = app
        self.tone = tone
        self.history = deque(maxlen=4)

        self.frame = ttk.Frame(parent, padding=(8, 8), style="Card.TFrame")
        self.frame.pack(fill="x", pady=4)

        self.freq_var = tk.StringVar(value=f"{tone.freq:.2f}")
        self.wave_var = tk.StringVar(value=tone.wave_type)
        self.vol_var = tk.DoubleVar(value=tone.volume * 100.0)

        self.frame.columnconfigure(5, weight=1)

        ttk.Label(self.frame, text="Hz").grid(row=0, column=0, sticky="w")
        self.freq_entry = ttk.Entry(self.frame, width=10, textvariable=self.freq_var)
        self.freq_entry.grid(row=0, column=1, padx=(4, 10), sticky="w")
        self.freq_entry.bind("<Return>", self.apply_ui_to_tone)
        self.freq_entry.bind("<FocusOut>", self.apply_ui_to_tone)

        ttk.Label(self.frame, text="Wave").grid(row=0, column=2, sticky="w")
        self.wave_combo = ttk.Combobox(
            self.frame,
            width=10,
            textvariable=self.wave_var,
            values=WAVE_TYPES,
            state="readonly",
        )
        self.wave_combo.grid(row=0, column=3, padx=(4, 10), sticky="w")
        self.wave_combo.bind("<<ComboboxSelected>>", self.apply_ui_to_tone)
        self.wave_combo.bind("<Key>", lambda e: "break")

        ttk.Label(self.frame, text="Volume").grid(row=0, column=4, sticky="w")
        self.vol_scale = ttk.Scale(
            self.frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.vol_var,
            command=self.on_volume_change,
        )
        self.vol_scale.grid(row=0, column=5, padx=(4, 10), sticky="ew")

        self.vol_label = ttk.Label(self.frame, text=f"{self.vol_var.get():.0f}%", style="Muted.TLabel")
        self.vol_label.grid(row=0, column=6, sticky="w")

        self.mute_btn = ttk.Button(self.frame, text="Mute", command=self.toggle_mute)
        self.mute_btn.grid(row=0, column=7, padx=(10, 6), sticky="e")

        self.reset_btn = ttk.Button(self.frame, text="Reset", command=self.reset_volume)
        self.reset_btn.grid(row=0, column=8, padx=(0, 6), sticky="e")

        self.remove_btn = ttk.Button(self.frame, text="Remove", command=self.remove)
        self.remove_btn.grid(row=0, column=9, sticky="e")

        self.preview = tk.Canvas(
            self.frame,
            width=240,
            height=54,
            bg=self.app.palette["canvas"],
            highlightthickness=1,
            highlightbackground=self.app.palette["edge"],
        )
        self.preview.grid(row=1, column=0, columnspan=10, pady=(8, 0), sticky="ew")

        if not self.app.show_waveforms_var.get():
            self.preview.grid_remove()

        self.update_mute_button()
        self.apply_ui_to_tone()

    def set_waveform_visible(self, visible: bool):
        if visible:
            self.preview.grid()
        else:
            self.preview.grid_remove()

    def update_mute_button(self):
        self.mute_btn.config(text="Unmute" if self.tone.muted else "Mute")

    def toggle_mute(self):
        with tone_lock:
            self.tone.muted = not self.tone.muted
        self.update_mute_button()

    def on_volume_change(self, _value=None):
        self.vol_label.config(text=f"{self.vol_var.get():.0f}%")
        self.apply_ui_to_tone()

    def reset_volume(self):
        self.vol_var.set(self.tone.initial_volume * 100.0)
        self.on_volume_change()

    def apply_ui_to_tone(self, event=None):
        try:
            freq = float(self.freq_var.get().strip())
            if freq <= 0:
                raise ValueError
        except ValueError:
            return

        wave_type = self.wave_var.get().strip().lower()
        if wave_type not in WAVE_TYPES:
            wave_type = "sine"

        volume = max(0.0, min(1.0, self.vol_var.get() / 100.0))

        with tone_lock:
            self.tone.freq = freq
            self.tone.wave_type = wave_type
            self.tone.volume = volume

    def update_theme(self):
        self.frame.configure(style="Card.TFrame")
        self.preview.configure(bg=self.app.palette["canvas"], highlightbackground=self.app.palette["edge"])

    def draw_preview(self):
        if not self.app.show_waveforms_var.get():
            return

        self.preview.delete("all")
        w = int(self.preview.winfo_width() or 240)
        h = int(self.preview.winfo_height() or 54)
        mid = h // 2

        with tone_lock:
            freq = self.tone.freq
            wave_type = self.tone.wave_type
            volume = self.tone.volume
            phase0 = self.tone.phase
            muted = self.tone.muted

        n = 192
        y = render_tone_samples(freq, wave_type, volume, phase0, n)
        self.history.append(y)

        self.preview.create_rectangle(0, 0, w, h, outline="", fill=self.app.palette["canvas"])
        self.preview.create_line(0, mid, w, mid, fill=self.app.palette["grid"])

        hist = list(self.history)
        for idx, frame in enumerate(hist):
            if muted:
                color = self.app.palette["muted"]
            else:
                color = self.app.palette["trace"][min(idx, len(self.app.palette["trace"]) - 1)]
            width = 1 if idx < len(hist) - 1 else 2
            points = samples_to_points(frame, w, h, y_scale=0.38)
            self.preview.create_line(*points, fill=color, width=width)

        self.preview.create_line(0, mid, w, mid, fill=self.app.palette["grid"])

    def remove(self):
        self.app.remove_tone(self.tone.tone_id)

# Main app
class ToneMixerApp:
    def update_tones_scrollregion(self, event=None):
        bbox = self.tones_canvas.bbox("all")
        if bbox:
            self.tones_canvas.configure(scrollregion=bbox)

    def _on_tones_canvas_configure(self, event):
        self.tones_canvas.itemconfigure(self.tones_window, width=event.width)

    def _is_descendant_of(self, widget, ancestor) -> bool:
        while widget is not None:
            if widget == ancestor:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_global_mousewheel(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return

        if self._is_descendant_of(widget, self.tones_frame):
            delta = 0
            if event.delta:
                delta = int(-1 * (event.delta / 120))
            if delta != 0:
                self.tones_canvas.yview_scroll(delta, "units")
                return "break"

    def __init__(self, root, audio_stream, pyaudio_instance):
        self.root = root
        self.stream = audio_stream
        self.p = pyaudio_instance
        self.theme_var = tk.BooleanVar(value=True)
        self.show_waveforms_var = tk.BooleanVar(value=True)
        self.palette = configure_theme(self.root, "dark")
        self.root.title("Tone Mixer Pro")
        self.root.option_add("*Font", "Arial 10")

        self.main = ttk.Frame(root, padding=12)
        self.main.pack(fill="both", expand=True)

        header = ttk.Frame(self.main)
        header.pack(fill="x", pady=(0, 10))

        self.title_label = tk.Label(
            header,
            text="Tone Mixer Pro",
            bg=self.palette["bg"],
            fg=self.palette["fg"],
            font=("Arial", 16, "bold"),
        )
        self.title_label.pack(side="left")

        self.subtitle_label = tk.Label(
            header,
            text="Wave mixer with live previews",
            bg=self.palette["bg"],
            fg=self.palette["muted"],
            font=("Arial", 10),
        )
        self.subtitle_label.pack(side="left", padx=(10, 0))

        controls = ttk.Frame(self.main)
        controls.pack(fill="x", pady=(0, 10))

        self.theme_toggle = ttk.Checkbutton(
            controls,
            text="Dark theme",
            variable=self.theme_var,
            command=self.toggle_theme,
        )
        self.theme_toggle.pack(side="left", padx=(0, 12))

        self.wave_toggle = ttk.Checkbutton(
            controls,
            text="Show waveforms",
            variable=self.show_waveforms_var,
            command=self.update_waveform_visibility,
        )
        self.wave_toggle.pack(side="left", padx=(0, 12))

        self.pause_btn = ttk.Button(controls, text="Pause all", command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=(0, 12))

        self.clear_btn = ttk.Button(controls, text="Clear all tones", command=self.clear_all_tones)
        self.clear_btn.pack(side="left")

        # Master volume
        master_frame = ttk.Frame(self.main)
        master_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(master_frame, text="Master volume").pack(side="left")
        self.master_vol_var = tk.DoubleVar(value=50.0)
        self.master_vol_scale = ttk.Scale(
            master_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.master_vol_var,
            command=self.on_master_volume_change,
        )
        self.master_vol_scale.pack(side="left", fill="x", expand=True, padx=(8, 8))
        self.master_vol_label = ttk.Label(master_frame, text="50%", style="Muted.TLabel")
        self.master_vol_label.pack(side="left")
        self.master_vol_reset_btn = ttk.Button(master_frame, text="Reset", command=self.reset_master_volume)
        self.master_vol_reset_btn.pack(side="left")

        # Mixed waveform zoom
        self.mix_zoom_var = tk.DoubleVar(value=1.0)

        add_frame = ttk.LabelFrame(self.main, text="Add tone", padding=10)
        add_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(add_frame, text="Frequency (Hz)").grid(row=0, column=0, sticky="w")
        self.new_freq_var = tk.StringVar(value="")
        self.new_freq_entry = ttk.Entry(add_frame, width=10, textvariable=self.new_freq_var)
        self.new_freq_entry.grid(row=0, column=1, padx=(4, 12), sticky="w")

        ttk.Label(add_frame, text="Wave").grid(row=0, column=2, sticky="w")
        self.new_wave_var = tk.StringVar(value="sine")
        self.new_wave_combo = ttk.Combobox(
            add_frame,
            width=10,
            textvariable=self.new_wave_var,
            values=WAVE_TYPES,
            state="readonly",
        )
        self.new_wave_combo.grid(row=0, column=3, padx=(4, 12), sticky="w")
        self.new_wave_combo.bind("<Key>", lambda e: "break")

        ttk.Label(add_frame, text="Volume").grid(row=0, column=4, sticky="w")
        self.new_vol_var = tk.DoubleVar(value=50.0)
        self.new_vol_scale = ttk.Scale(
            add_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.new_vol_var,
            command=self.on_new_volume_change,
        )
        self.new_vol_scale.grid(row=0, column=5, padx=(4, 10), sticky="ew")

        self.new_vol_label = ttk.Label(add_frame, text="50%", style="Muted.TLabel")
        self.new_vol_label.grid(row=0, column=6, sticky="w")

        self.new_vol_reset_btn = ttk.Button(add_frame, text="Reset", command=self.reset_new_volume)
        self.new_vol_reset_btn.grid(row=0, column=7, padx=(10, 6), sticky="e")

        self.add_btn = ttk.Button(add_frame, text="Add tone", command=self.add_tone)
        self.add_btn.grid(row=0, column=8, sticky="e")

        add_frame.columnconfigure(5, weight=1)
        self.new_freq_entry.focus_set()

        self.mix_frame = ttk.LabelFrame(self.main, text="Mixed waveform", padding=10)
        self.mix_frame.pack(fill="x", pady=(0, 10))

        mix_zoom_controls = ttk.Frame(self.mix_frame)
        mix_zoom_controls.pack(fill="x", pady=(0, 8))

        ttk.Label(mix_zoom_controls, text="Zoom").pack(side="left")
        self.mix_zoom_scale = ttk.Scale(
            mix_zoom_controls,
            from_=0.5,
            to=3.0,
            orient="horizontal",
            variable=self.mix_zoom_var,
        )
        self.mix_zoom_scale.pack(side="left", fill="x", expand=True, padx=(8, 8))
        self.mix_zoom_reset_btn = ttk.Button(mix_zoom_controls, text="Reset zoom", command=self.reset_mix_zoom)
        self.mix_zoom_reset_btn.pack(side="left")

        self.mix_canvas = tk.Canvas(
            self.mix_frame,
            width=760,
            height=110,
            bg=self.palette["canvas"],
            highlightthickness=1,
            highlightbackground=self.palette["edge"],
        )
        self.mix_canvas.pack(fill="x")

        self.tones_frame = ttk.LabelFrame(self.main, text="Active tones", padding=10)
        self.tones_frame.pack(fill="both", expand=True)

        self.tones_canvas = tk.Canvas(
            self.tones_frame,
            bg=self.palette["bg"],
            highlightthickness=0,
        )
        self.tones_scrollbar = ttk.Scrollbar(self.tones_frame, orient="vertical", command=self.tones_canvas.yview)
        self.tones_canvas.configure(yscrollcommand=self.tones_scrollbar.set)

        self.tones_scrollbar.pack(side="right", fill="y")
        self.tones_canvas.pack(side="left", fill="both", expand=True)

        self.tones_container = ttk.Frame(self.tones_canvas)
        self.tones_window = self.tones_canvas.create_window((0, 0), window=self.tones_container, anchor="nw")

        self.tones_container.bind("<Configure>", self.update_tones_scrollregion)
        self.tones_canvas.bind("<Configure>", self._on_tones_canvas_configure)

        self.empty_label = ttk.Label(
            self.tones_container,
            text="No tones yet. Add one above.",
            style="Muted.TLabel",
        )
        self.empty_label.pack(anchor="center", pady=20)

        bottom = ttk.Frame(self.main)
        bottom.pack(fill="x", pady=(10, 0))

        self.quit_btn = ttk.Button(bottom, text="Quit", command=self.on_quit)
        self.quit_btn.pack(side="right")

        self.rows: dict[int, ToneRow] = {}
        self.mix_history = deque(maxlen=4)

        self.root.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
        self.root.after(60, self.refresh_visuals)

    def on_master_volume_change(self, _value=None):
        global master_volume
        self.master_vol_label.config(text=f"{self.master_vol_var.get():.0f}%")
        master_volume = self.master_vol_var.get() / 100.0

    def reset_master_volume(self):
        self.master_vol_var.set(50.0)
        self.on_master_volume_change()

    def reset_mix_zoom(self):
        self.mix_zoom_var.set(1.0)

    def on_new_volume_change(self, _value=None):
        self.new_vol_label.config(text=f"{self.new_vol_var.get():.0f}%")

    def reset_new_volume(self):
        self.new_vol_var.set(50.0)
        self.on_new_volume_change()

    def toggle_theme(self):
        self.palette = configure_theme(self.root, "dark" if self.theme_var.get() else "light")
        self.theme_toggle.config(text="Dark theme" if self.theme_var.get() else "Light theme")

        self.title_label.config(bg=self.palette["bg"], fg=self.palette["fg"])
        self.subtitle_label.config(bg=self.palette["bg"], fg=self.palette["muted"])

        self.mix_canvas.configure(bg=self.palette["canvas"], highlightbackground=self.palette["edge"])
        self.tones_canvas.configure(bg=self.palette["bg"])

        for row in self.rows.values():
            row.update_theme()

        self.draw_mix_preview()

    def toggle_pause(self):
        global audio_paused
        audio_paused = not audio_paused
        self.pause_btn.config(text="Resume all" if audio_paused else "Pause all")

    def update_waveform_visibility(self):
        visible = self.show_waveforms_var.get()

        if visible:
            self.mix_frame.pack(fill="x", pady=(0, 10), before=self.tones_frame)
        else:
            self.mix_frame.pack_forget()

        for row in self.rows.values():
            row.set_waveform_visible(visible)

    def add_tone(self):
        global next_tone_id

        try:
            freq = float(self.new_freq_var.get().strip())
            if freq <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid frequency", "Enter a positive frequency in Hz.")
            return

        wave_type = self.new_wave_var.get().strip().lower()
        if wave_type not in WAVE_TYPES:
            wave_type = "sine"

        volume = max(0.0, min(1.0, self.new_vol_var.get() / 100.0))

        with tone_lock:
            tone = Tone(
                tone_id=next_tone_id,
                freq=freq,
                wave_type=wave_type,
                volume=volume,
                initial_volume=volume,
            )
            tones.append(tone)
            next_tone_id += 1

        if not self.rows:
            self.empty_label.pack_forget()

        row = ToneRow(self.tones_container, self, tone)
        tone.row = row
        self.rows[tone.tone_id] = row
        self.update_tones_scrollregion()

        self.new_freq_var.set("")
        self.new_freq_entry.focus_set()

    def remove_tone(self, tone_id: int):
        with tone_lock:
            for i, tone in enumerate(tones):
                if tone.tone_id == tone_id:
                    tones.pop(i)
                    break
            else:
                return

        row = self.rows.pop(tone_id, None)
        if row is not None:
            row.frame.destroy()

        if not self.rows:
            self.empty_label.pack(anchor="center", pady=20)
            self.update_tones_scrollregion()

    def clear_all_tones(self):
        with tone_lock:
            tones.clear()

        for row in list(self.rows.values()):
            row.frame.destroy()

        self.rows.clear()
        self.mix_history.clear()
        self.empty_label.pack(anchor="center", pady=20)
        self.tones_canvas.yview_moveto(0)
        self.update_tones_scrollregion()
        self.tones_canvas.yview_moveto(0)
        self.draw_mix_preview()

    def draw_mix_preview(self):
        if not self.show_waveforms_var.get():
            return

        self.mix_canvas.delete("all")
        w = int(self.mix_canvas.winfo_width() or 760)
        h = int(self.mix_canvas.winfo_height() or 110)
        mid = h // 2

        with tone_lock:
            snapshot = [
                (t.tone_id, t.freq, t.wave_type, t.volume, t.phase, t.muted)
                for t in tones
            ]

        self.mix_canvas.create_rectangle(0, 0, w, h, outline="", fill=self.palette["canvas"])
        self.mix_canvas.create_line(0, mid, w, mid, fill=self.palette["grid"])

        if not snapshot:
            return

        n = 384
        y = render_mix_samples(snapshot, n) * self.mix_zoom_var.get()
        self.mix_history.append(y)

        hist = list(self.mix_history)
        for idx, frame in enumerate(hist):
            color = self.palette["trace"][min(idx, len(self.palette["trace"]) - 1)]
            width = 1 if idx < len(hist) - 1 else 2
            points = samples_to_points(frame, w, h, y_scale=0.40)
            self.mix_canvas.create_line(*points, fill=color, width=width)

        self.mix_canvas.create_line(0, mid, w, mid, fill=self.palette["grid"])

    def refresh_visuals(self):
        if self.show_waveforms_var.get():
            with tone_lock:
                rows = list(self.rows.values())

            for row in rows:
                row.draw_preview()

            self.draw_mix_preview()

        self.root.after(60, self.refresh_visuals)

    def on_quit(self):
        global stream_running
        if messagebox.askokcancel("Quit", "Stop audio and exit?"):
            stream_running = False
            try:
                self.stream.stop_stream()
                self.stream.close()
            finally:
                self.p.terminate()
                self.root.destroy()

# Main
def main():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        frames_per_buffer=BUFFER_SIZE,
        output=True,
        stream_callback=audio_callback,
    )
    stream.start_stream()

    root = tk.Tk()
    app = ToneMixerApp(root, stream, p)
    root.protocol("WM_DELETE_WINDOW", app.on_quit)
    root.mainloop()


if __name__ == "__main__":
    main()
