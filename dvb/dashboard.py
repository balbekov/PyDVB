"""
DVB-T Audio Mode Debug Dashboard

Real-time Rich console dashboard for debugging DVB-T audio transmission/reception.
Shows signal quality, FEC performance, and transport stream statistics.

Usage:
    python -m dvb audio-rx-debug -d 30
    python -m dvb audio-rx-debug -i recording.wav
"""

import time
import threading
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

import numpy as np

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich import box

from .stats import StatsCollector, ImageStats

if TYPE_CHECKING:
    from .AudioInput import AcousticDVBTReceiver

# Try to import PIL for image preview
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Unicode block characters for visualizations
BLOCKS = " ▁▂▃▄▅▆▇█"
PHASE_CHARS = "→↗↑↖←↙↓↘"  # 8 directions for phase display


class DVBDashboard:
    """
    Real-time Rich console dashboard for DVB-T audio debugging.
    
    Displays:
    - Audio input status and levels
    - DVB-T modulation parameters
    - Signal quality (SNR, CFO, EVM)
    - I/Q constellation diagram
    - FEC statistics (RS errors, BER)
    - Transport stream throughput
    - OFDM symbol power sparkline
    - Rolling log messages
    
    Example:
        >>> from dvb.AudioInput import AcousticDVBTReceiver
        >>> from dvb.dashboard import DVBDashboard
        >>> 
        >>> rx = AcousticDVBTReceiver()
        >>> with DVBDashboard(rx) as dash:
        ...     ts_data, stats = rx.receive(30.0)
    """
    
    def __init__(self, 
                 receiver: Optional['AcousticDVBTReceiver'] = None,
                 refresh_rate: float = 10.0,
                 stats: Optional[StatsCollector] = None):
        """
        Initialize dashboard.
        
        Args:
            receiver: AcousticDVBTReceiver instance to monitor
            refresh_rate: Dashboard refresh rate in Hz (default 10)
            stats: Optional external StatsCollector (creates one if None)
        """
        self.receiver = receiver
        self.refresh_rate = refresh_rate
        self.stats = stats or StatsCollector()
        
        self.console = Console()
        self._live: Optional[Live] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Color theme
        self.theme = {
            'title': Style(color="bright_cyan", bold=True),
            'label': Style(color="bright_white"),
            'value': Style(color="green"),
            'value_warn': Style(color="yellow"),
            'value_error': Style(color="red"),
            'bar_good': Style(color="green"),
            'bar_warn': Style(color="yellow"),
            'bar_error': Style(color="red"),
            'dim': Style(color="bright_black"),
            'accent': Style(color="cyan"),
        }
    
    def _make_bar(self, value: float, max_value: float, width: int = 14,
                  thresholds: tuple = (0.3, 0.7)) -> Text:
        """Create a colored progress bar."""
        ratio = min(1.0, max(0.0, value / max_value)) if max_value > 0 else 0
        filled = int(ratio * width)
        
        # Choose color based on thresholds
        if ratio < thresholds[0]:
            style = self.theme['bar_error']
        elif ratio < thresholds[1]:
            style = self.theme['bar_warn']
        else:
            style = self.theme['bar_good']
        
        bar = Text()
        bar.append("█" * filled, style=style)
        bar.append("░" * (width - filled), style=self.theme['dim'])
        return bar
    
    def _make_sparkline(self, values: List[float], width: int = 40) -> Text:
        """Create a sparkline from values."""
        if not values:
            return Text("─" * width, style=self.theme['dim'])
        
        # Normalize to 0-8 range for block characters
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1.0
        
        # Sample or pad to width
        if len(values) >= width:
            sampled = values[-width:]
        else:
            sampled = values
        
        text = Text()
        for v in sampled:
            idx = int((v - min_val) / range_val * 7.99)
            idx = max(0, min(8, idx))
            text.append(BLOCKS[idx], style=self.theme['accent'])
        
        # Pad if needed
        if len(sampled) < width:
            text.append("─" * (width - len(sampled)), style=self.theme['dim'])
        
        return text
    
    def _make_constellation(self, iq_samples: np.ndarray, 
                            size: int = 15) -> Text:
        """
        Create ASCII I/Q constellation diagram.
        
        Args:
            iq_samples: Complex I/Q samples
            size: Grid size (size x size)
            
        Returns:
            Text renderable with constellation plot
        """
        # Create empty grid
        grid = [[' ' for _ in range(size)] for _ in range(size)]
        
        if len(iq_samples) == 0:
            # Empty constellation
            lines = Text()
            for row in grid:
                lines.append(''.join(row) + '\n', style=self.theme['dim'])
            return lines
        
        # Normalize I/Q to grid coordinates
        max_amp = np.max(np.abs(iq_samples)) * 1.2
        if max_amp == 0:
            max_amp = 1.0
        
        # Count samples per grid cell
        counts = np.zeros((size, size), dtype=int)
        
        for sample in iq_samples:
            i_norm = (np.real(sample) / max_amp + 1) / 2  # 0 to 1
            q_norm = (np.imag(sample) / max_amp + 1) / 2  # 0 to 1
            
            x = int(i_norm * (size - 1))
            y = int((1 - q_norm) * (size - 1))  # Flip Y axis
            
            x = max(0, min(size - 1, x))
            y = max(0, min(size - 1, y))
            
            counts[y, x] += 1
        
        # Convert counts to characters
        max_count = np.max(counts)
        if max_count > 0:
            for y in range(size):
                for x in range(size):
                    if counts[y, x] > 0:
                        # Intensity based on count
                        intensity = counts[y, x] / max_count
                        if intensity > 0.7:
                            grid[y][x] = '◉'
                        elif intensity > 0.3:
                            grid[y][x] = '●'
                        else:
                            grid[y][x] = '·'
        
        # Add center crosshair
        mid = size // 2
        if grid[mid][mid] == ' ':
            grid[mid][mid] = '+'
        
        # Build text
        lines = Text()
        for row in grid:
            lines.append(''.join(row) + '\n', style=self.theme['accent'])
        
        return lines
    
    def _make_waveform(self, levels: List[float], width: int = 30, height: int = 4) -> Text:
        """
        Create a multi-line waveform display using block characters.
        
        Shows a scrolling waveform like an oscilloscope.
        """
        if not levels:
            text = Text()
            for _ in range(height):
                text.append("─" * width + "\n", style=self.theme['dim'])
            return text
        
        # Take the most recent samples
        samples = levels[-width:] if len(levels) >= width else levels
        
        # Normalize to 0-1 range
        max_val = max(samples) if samples else 1.0
        if max_val == 0:
            max_val = 1.0
        normalized = [min(1.0, s / max_val) for s in samples]
        
        # Build the waveform grid
        # Use vertical block characters: ▁▂▃▄▅▆▇█
        grid = []
        for row in range(height):
            row_threshold = 1.0 - (row + 0.5) / height
            line = Text()
            for val in normalized:
                if val >= row_threshold + 0.125:
                    line.append("█", style=Style(color="green"))
                elif val >= row_threshold:
                    line.append("▄", style=Style(color="green"))
                elif val >= row_threshold - 0.125:
                    line.append("▀", style=Style(color="bright_black"))
                else:
                    line.append(" ")
            # Pad if needed
            if len(normalized) < width:
                line.append(" " * (width - len(normalized)), style=self.theme['dim'])
            line.append("\n")
            grid.append(line)
        
        result = Text()
        for line in grid:
            result.append_text(line)
        return result
    
    def _make_level_meter(self, level: float, peak: float, width: int = 20) -> Text:
        """Create a VU-style level meter with peak hold."""
        meter = Text()
        
        # Normalize (assuming 0-1 range, with typical levels around 0.3)
        level_pos = int(min(1.0, level * 2) * width)
        peak_pos = int(min(1.0, peak * 2) * width)
        
        for i in range(width):
            if i < level_pos:
                if i < width * 0.6:
                    meter.append("█", style=Style(color="green"))
                elif i < width * 0.8:
                    meter.append("█", style=Style(color="yellow"))
                else:
                    meter.append("█", style=Style(color="red"))
            elif i == peak_pos and peak_pos > 0:
                meter.append("│", style=Style(color="white", bold=True))
            else:
                meter.append("░", style=self.theme['dim'])
        
        return meter
    
    def _audio_input_panel(self) -> Panel:
        """Create audio input status panel with waveform."""
        snap = self.stats.get_snapshot()
        audio = snap['audio']
        
        content = Text()
        
        # Status line
        status = "● LIVE" if audio.is_streaming else "○ Idle"
        status_style = Style(color="green", bold=True) if audio.is_streaming else self.theme['dim']
        content.append(status, style=status_style)
        content.append(f"  {audio.sample_rate/1000:.0f}kHz  ", style=self.theme['dim'])
        content.append(f"fc={audio.carrier_freq/1000:.0f}kHz\n", style=self.theme['dim'])
        
        # Level meter
        content.append("Level: ", style=self.theme['label'])
        level_meter = self._make_level_meter(audio.rms_level, audio.peak_level, width=20)
        content.append_text(level_meter)
        content.append(f" {audio.rms_level:.2f}\n", style=self.theme['dim'])
        
        # Waveform display
        content.append("Waveform:\n", style=self.theme['label'])
        levels = self.stats.get_level_history()
        waveform = self._make_waveform(levels, width=28, height=3)
        content.append_text(waveform)
        
        return Panel(
            content,
            title="[bold cyan]AUDIO INPUT[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    
    def _dvbt_params_panel(self) -> Panel:
        """Create DVB-T parameters panel."""
        snap = self.stats.get_snapshot()
        dvbt = snap['dvbt']
        
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Label", style=self.theme['label'])
        table.add_column("Value", style=self.theme['value'])
        
        # Mode with FFT size
        mode_desc = f"{dvbt.mode}"
        if dvbt.mode == 'audio':
            mode_desc += f" ({dvbt.fft_size}-pt FFT)"
        table.add_row("Mode:", mode_desc)
        
        # Constellation
        table.add_row("Constellation:", dvbt.constellation)
        
        # Code rate
        table.add_row("Code Rate:", dvbt.code_rate)
        
        # Guard interval
        table.add_row("Guard:", dvbt.guard_interval)
        
        # Data carriers
        table.add_row("Carriers:", f"{dvbt.data_carriers}")
        
        # Net data rate
        if dvbt.data_rate > 0:
            rate_str = f"{dvbt.data_rate / 1000:.1f} kbps"
        else:
            rate_str = "—"
        table.add_row("Data Rate:", rate_str)
        
        return Panel(
            table,
            title="[bold cyan]DVB-T PARAMETERS[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    
    def _signal_quality_panel(self) -> Panel:
        """Create signal quality metrics panel."""
        snap = self.stats.get_snapshot()
        signal = snap['signal']
        
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Label", style=self.theme['label'], width=6)
        table.add_column("Value", width=10)
        table.add_column("Bar", width=14)
        
        # SNR with bar
        snr_bar = self._make_bar(signal.snr_db, 30.0, width=14, thresholds=(0.3, 0.6))
        snr_style = self.theme['value'] if signal.snr_db > 10 else self.theme['value_warn']
        table.add_row("SNR:", Text(f"{signal.snr_db:.1f} dB", style=snr_style), snr_bar)
        
        # CFO
        cfo_abs = abs(signal.cfo_hz)
        cfo_style = self.theme['value'] if cfo_abs < 100 else self.theme['value_warn']
        cfo_sign = "+" if signal.cfo_hz >= 0 else ""
        table.add_row("CFO:", Text(f"{cfo_sign}{signal.cfo_hz:.1f} Hz", style=cfo_style), Text(""))
        
        # EVM with bar (lower is better)
        evm_bar = self._make_bar(100 - signal.evm_percent, 100.0, width=14, thresholds=(0.5, 0.8))
        evm_style = self.theme['value'] if signal.evm_percent < 15 else self.theme['value_warn']
        table.add_row("EVM:", Text(f"{signal.evm_percent:.1f}%", style=evm_style), evm_bar)
        
        # MER
        mer_style = self.theme['value'] if signal.mer_db > 15 else self.theme['value_warn']
        table.add_row("MER:", Text(f"{signal.mer_db:.0f} dB", style=mer_style), Text(""))
        
        # Phase indicator
        phase_idx = int((signal.phase_deg % 360) / 45) % 8
        phase_char = PHASE_CHARS[phase_idx]
        table.add_row("Phase:", Text(f"{phase_char} ({signal.phase_deg:.0f}°)", style=self.theme['accent']), Text(""))
        
        # Signal present indicator
        if signal.signal_present:
            sig_text = Text("● LOCKED", style=Style(color="green", bold=True))
        else:
            sig_text = Text("○ Searching", style=self.theme['dim'])
        table.add_row("", sig_text, Text(""))
        
        return Panel(
            table,
            title="[bold cyan]SIGNAL QUALITY[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    
    def _constellation_panel(self) -> Panel:
        """Create I/Q constellation diagram panel."""
        iq = self.stats.get_iq_samples()
        const_diagram = self._make_constellation(iq, size=13)
        
        return Panel(
            const_diagram,
            title="[bold cyan]I/Q CONSTELLATION[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    
    def _fec_panel(self) -> Panel:
        """Create FEC statistics panel."""
        snap = self.stats.get_snapshot()
        fec = snap['fec']
        
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Label", style=self.theme['label'])
        table.add_column("Value", style=self.theme['value'])
        
        # RS corrected
        rs_style = self.theme['value'] if fec.rs_corrected == 0 else self.theme['value_warn']
        table.add_row("RS Corrected:", Text(f"{fec.rs_corrected:,} errors", style=rs_style))
        
        # RS uncorrectable
        rs_unc_style = self.theme['value'] if fec.rs_uncorrectable == 0 else self.theme['value_error']
        table.add_row("RS Uncorrect:", Text(f"{fec.rs_uncorrectable:,} blocks", style=rs_unc_style))
        
        # Pre-FEC BER
        if fec.ber_pre_fec > 0:
            ber_pre = f"{fec.ber_pre_fec:.2e}"
        else:
            ber_pre = "0"
        table.add_row("BER (pre):", ber_pre)
        
        # Post-FEC BER
        if fec.ber_post_fec > 0:
            ber_post = f"{fec.ber_post_fec:.2e}"
            ber_style = self.theme['value_error']
        else:
            ber_post = "0"
            ber_style = self.theme['value']
        table.add_row("BER (post):", Text(ber_post, style=ber_style))
        
        # Viterbi errors
        table.add_row("Viterbi:", f"{fec.viterbi_errors:,}")
        
        return Panel(
            table,
            title="[bold cyan]FEC PERFORMANCE[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    
    def _transport_panel(self) -> Panel:
        """Create transport stream statistics panel."""
        snap = self.stats.get_snapshot()
        ts = snap['transport']
        
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Label", style=self.theme['label'])
        table.add_column("Value", style=self.theme['value'])
        
        # Packets
        table.add_row("Packets:", f"{ts.packets_received:,}")
        
        # Bytes
        if ts.bytes_received >= 1024 * 1024:
            bytes_str = f"{ts.bytes_received / 1024 / 1024:.2f} MB"
        elif ts.bytes_received >= 1024:
            bytes_str = f"{ts.bytes_received / 1024:.1f} KB"
        else:
            bytes_str = f"{ts.bytes_received:,} B"
        table.add_row("Bytes:", bytes_str)
        
        # Throughput
        if ts.throughput_bps >= 1000:
            tp_str = f"{ts.throughput_bps / 1000:.1f} kbps"
        else:
            tp_str = f"{ts.throughput_bps:.0f} bps"
        table.add_row("Throughput:", tp_str)
        
        # Duration
        mins = int(ts.duration_sec // 60)
        secs = ts.duration_sec % 60
        if mins > 0:
            dur_str = f"{mins}m {secs:.1f}s"
        else:
            dur_str = f"{secs:.1f}s"
        table.add_row("Duration:", dur_str)
        
        # Symbols
        table.add_row("Symbols:", f"{ts.symbols_processed:,}")
        
        # Frames
        table.add_row("Frames:", f"{ts.frames_processed:,}")
        
        return Panel(
            table,
            title="[bold cyan]TRANSPORT STREAM[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    
    def _image_to_ascii(self, image_data: bytes, width: int = 30, height: int = 15) -> Text:
        """
        Convert image data to ASCII art using block characters.
        
        Uses Unicode block characters to create a grayscale preview.
        """
        if not HAS_PIL or not image_data:
            return Text("No image data", style=self.theme['dim'])
        
        try:
            import io
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale and resize
            img = img.convert('L')
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Block characters from dark to light
            blocks = " ░▒▓█"
            
            text = Text()
            pixels = list(img.getdata())
            
            for y in range(height):
                for x in range(width):
                    pixel = pixels[y * width + x]
                    # Map 0-255 to block index
                    idx = int(pixel / 256 * len(blocks))
                    idx = min(len(blocks) - 1, idx)
                    text.append(blocks[idx], style=self.theme['accent'])
                text.append('\n')
            
            return text
            
        except Exception as e:
            return Text(f"Preview error: {e}", style=self.theme['dim'])
    
    def _image_panel(self) -> Panel:
        """Create image reception panel with preview."""
        snap = self.stats.get_snapshot()
        img = snap['image']
        
        content = Text()
        
        if not img.found_header:
            content.append("Waiting for image data...\n", style=self.theme['dim'])
            content.append("\n" * 10)  # Placeholder space
        else:
            # Image info
            content.append("Format: ", style=self.theme['label'])
            content.append(f"{img.image_format.upper()}\n", style=self.theme['value'])
            
            content.append("Size: ", style=self.theme['label'])
            content.append(f"{img.width}×{img.height}\n", style=self.theme['value'])
            
            # Progress bar
            content.append("Progress: ", style=self.theme['label'])
            progress_bar = self._make_bar(img.progress_percent, 100.0, width=20, 
                                          thresholds=(0.3, 0.9))
            content.append_text(progress_bar)
            content.append(f" {img.progress_percent:.1f}%\n", style=self.theme['value'])
            
            # Size received
            if img.expected_size > 0:
                content.append("Received: ", style=self.theme['label'])
                content.append(f"{img.received_size:,}/{img.expected_size:,} bytes\n", 
                             style=self.theme['value'])
            
            # CRC status
            if img.progress_percent >= 100:
                if img.crc_valid:
                    content.append("CRC: ", style=self.theme['label'])
                    content.append("✓ Valid\n", style=Style(color="green", bold=True))
                else:
                    content.append("CRC: ", style=self.theme['label'])
                    content.append("✗ Invalid\n", style=Style(color="red", bold=True))
            
            content.append("\n")
            
            # Image preview
            if img.image_data:
                content.append("Preview:\n", style=self.theme['label'])
                ascii_preview = self._image_to_ascii(img.image_data, width=28, height=12)
                content.append_text(ascii_preview)
            elif img.progress_percent > 0:
                # Show partial loading animation
                content.append("Loading preview", style=self.theme['dim'])
                dots = "." * (int(time.time() * 2) % 4)
                content.append(dots + "\n", style=self.theme['dim'])
                content.append("\n" * 8)
        
        return Panel(
            content,
            title="[bold cyan]IMAGE PREVIEW[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    
    def _ofdm_sparkline_panel(self) -> Panel:
        """Create OFDM symbol power sparkline panel."""
        powers = self.stats.get_symbol_power()
        sparkline = self._make_sparkline(powers, width=60)
        
        snap = self.stats.get_snapshot()
        ts = snap['transport']
        
        # Add stats
        content = Text()
        content.append_text(sparkline)
        content.append(f"  ({ts.symbols_processed} symbols, {ts.frames_processed} frames)", 
                      style=self.theme['dim'])
        
        return Panel(
            content,
            title="[bold cyan]OFDM SYMBOLS[/]",
            border_style="cyan",
            box=box.ROUNDED,
            height=3,
        )
    
    def _log_panel(self) -> Panel:
        """Create scrolling log panel."""
        messages = self.stats.get_log_messages(6)
        
        log_text = Text()
        for timestamp, level, message in messages:
            # Format timestamp
            dt = datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}"
            
            # Level colors
            if level == "ERROR":
                level_style = Style(color="red", bold=True)
            elif level == "WARN":
                level_style = Style(color="yellow")
            else:
                level_style = Style(color="green")
            
            log_text.append(f"{time_str}  ", style=self.theme['dim'])
            log_text.append(f"{level:5}  ", style=level_style)
            log_text.append(f"{message}\n", style=self.theme['label'])
        
        if not messages:
            log_text.append("Waiting for events...", style=self.theme['dim'])
        
        return Panel(
            log_text,
            title="[bold cyan]LOG[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    
    def _make_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        # Main structure - vertical split
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="top_row", size=12),
            Layout(name="middle_row", size=14),
            Layout(name="bottom_row"),
        )
        
        # Top row: Audio + DVB-T + FEC
        layout["top_row"].split_row(
            Layout(name="audio"),
            Layout(name="dvbt"),
            Layout(name="fec"),
        )
        
        # Middle row: Signal + Constellation + Transport
        layout["middle_row"].split_row(
            Layout(name="signal"),
            Layout(name="constellation"),
            Layout(name="transport"),
        )
        
        # Bottom row: OFDM + Image + Log
        layout["bottom_row"].split_row(
            Layout(name="ofdm_log", ratio=2),
            Layout(name="image", ratio=1),
        )
        
        # Split ofdm_log into ofdm sparkline and log
        layout["ofdm_log"].split_column(
            Layout(name="ofdm", size=4),
            Layout(name="log"),
        )
        
        return layout
    
    def _update_layout(self, layout: Layout) -> None:
        """Update all panels in the layout."""
        # Header
        header_text = Text()
        header_text.append("  PyDVB Audio Mode ", style=Style(color="bright_cyan", bold=True))
        header_text.append("─ Real-Time Debug Dashboard", style=self.theme['dim'])
        
        snap = self.stats.get_snapshot()
        if snap['signal'].signal_present:
            header_text.append("  ● SIGNAL", style=Style(color="green", bold=True))
        else:
            header_text.append("  ○ NO SIGNAL", style=self.theme['dim'])
        
        # Show image progress if receiving
        if snap['image'].found_header:
            header_text.append(f"  📷 {snap['image'].progress_percent:.0f}%", 
                             style=Style(color="yellow"))
        
        layout["header"].update(Panel(header_text, box=box.ROUNDED, border_style="bright_cyan"))
        
        # Left column
        layout["audio"].update(self._audio_input_panel())
        layout["signal"].update(self._signal_quality_panel())
        
        # Middle column
        layout["dvbt"].update(self._dvbt_params_panel())
        layout["constellation"].update(self._constellation_panel())
        
        # Right column
        layout["fec"].update(self._fec_panel())
        layout["transport"].update(self._transport_panel())
        layout["image"].update(self._image_panel())
        
        # Bottom panels
        layout["ofdm"].update(self._ofdm_sparkline_panel())
        layout["log"].update(self._log_panel())
    
    def start(self) -> None:
        """Start the dashboard."""
        if self._running:
            return
        
        self._running = True
        self.stats.start()
        
        # Initialize DVB-T params from receiver if available
        if self.receiver is not None:
            self.stats.update_audio(
                sample_rate=self.receiver.audio_sample_rate,
                carrier_freq=self.receiver.carrier_freq,
                audio_library=self.receiver.audio_input._audio_lib or "none",
            )
            self.stats.update_dvbt_params(
                mode=self.receiver.demodulator.mode,
                constellation=self.receiver.demodulator.constellation,
                code_rate=self.receiver.demodulator.code_rate,
                guard_interval=self.receiver.demodulator.guard_interval,
                bandwidth=self.receiver.demodulator.bandwidth,
                fft_size=getattr(self.receiver.demodulator, 'fft_size', 128),
                sample_rate=self.receiver.dvbt_sample_rate,
            )
        
        self._layout = self._make_layout()
        self._update_layout(self._layout)
    
    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
    
    def refresh(self) -> None:
        """Refresh the dashboard display."""
        if self._running and self._layout is not None:
            self._update_layout(self._layout)
    
    def get_renderable(self):
        """Get the current layout for Live context."""
        if self._layout is None:
            self._layout = self._make_layout()
        self._update_layout(self._layout)
        return self._layout
    
    def log(self, level: str, message: str) -> None:
        """Add a log message."""
        self.stats.log(level, message)
    
    def __enter__(self) -> 'DVBDashboard':
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.stop()
        return False


def run_debug_dashboard(
    input_file: Optional[str] = None,
    duration: float = 30.0,
    sample_rate: int = 48000,
    carrier_freq: float = 5000,
    output_file: Optional[str] = None,
    output_image: Optional[str] = None,
    save_audio: Optional[str] = None,
) -> tuple:
    """
    Run the debug dashboard for audio reception.
    
    Args:
        input_file: Input WAV file (None for live microphone)
        duration: Recording duration in seconds (for live mode)
        sample_rate: Audio sample rate
        carrier_freq: Carrier frequency
        output_file: Output file for recovered transport stream
        output_image: Output file for recovered image (auto-detected extension)
        save_audio: Save input audio to WAV file
        
    Returns:
        Tuple of (recovered_data, stats)
    """
    from .AudioInput import AcousticDVBTReceiver
    import threading
    import shutil
    
    # Create stats collector first
    stats = StatsCollector()
    
    # Create receiver with stats collector
    rx = AcousticDVBTReceiver(
        audio_sample_rate=sample_rate,
        carrier_freq=carrier_freq,
        stats_collector=stats,
    )
    
    # Create dashboard
    dashboard = DVBDashboard(rx, stats=stats, refresh_rate=10.0)
    
    # Results container for thread
    result = {'ts_data': b'', 'rx_stats': {}, 'done': False, 'error': None}
    
    def process_file():
        """Process file in background thread."""
        try:
            if input_file:
                result['ts_data'], result['rx_stats'] = rx.receive_file(input_file)
            else:
                # Record audio (optionally saving raw audio)
                iq_samples = rx.audio_input.record(duration, save_raw_audio=save_audio)
                
                # Resample and process
                iq_resampled = rx._resample_iq(
                    iq_samples,
                    rx.audio_sample_rate,
                    rx.dvbt_sample_rate
                )
                result['ts_data'], result['rx_stats'] = rx._demodulate_with_stats(iq_resampled)
        except Exception as e:
            import traceback
            traceback.print_exc()
            result['error'] = e
        finally:
            result['done'] = True
    
    console = Console()
    ts_data = b''
    rx_stats = {}
    
    try:
        dashboard.start()
        dashboard.log("INFO", "Dashboard started")
        
        # Update DVB-T params display
        stats.update_dvbt_params(
            mode=rx.demodulator.mode,
            constellation=rx.demodulator.constellation,
            code_rate=rx.demodulator.code_rate,
            guard_interval=rx.demodulator.guard_interval,
            fft_size=getattr(rx.demodulator, 'fft_size', 128),
            data_carriers=96,  # Audio mode
            sample_rate=rx.dvbt_sample_rate,
        )
        
        if input_file:
            dashboard.log("INFO", f"Processing: {input_file}")
            # Copy input file if save_audio requested
            if save_audio:
                shutil.copy(input_file, save_audio)
                dashboard.log("INFO", f"Copied input to: {save_audio}")
        else:
            dashboard.log("INFO", f"Recording for {duration:.1f} seconds...")
        
        stats.update_audio(is_streaming=True)
        
        # Run live dashboard while processing
        with Live(dashboard.get_renderable(), console=console, refresh_per_second=20, screen=True) as live:
            # Show initial dashboard state before starting processing
            live.update(dashboard.get_renderable())
            
            # Start processing in background thread
            process_thread = threading.Thread(target=process_file, daemon=True)
            process_thread.start()
            
            # Update display while processing
            while not result['done']:
                live.update(dashboard.get_renderable())
                time.sleep(0.05)  # 20 FPS updates
            
            # Processing done - update final state
            ts_data = result['ts_data']
            rx_stats = result['rx_stats']
            
            # Check for errors
            if result['error']:
                dashboard.log("ERROR", str(result['error']))
            else:
                # Update final stats
                stats.update_signal(
                    snr_db=rx_stats.get('snr_db', 0),
                    cfo_hz=rx_stats.get('cfo_hz', 0),
                    signal_present=rx_stats.get('packets_recovered', 0) > 0,
                )
                
                dashboard.log("INFO", f"Recovered {len(ts_data):,} bytes ({rx_stats.get('packets_recovered', 0)} packets)")
            
            stats.update_audio(is_streaming=False)
            
            # Save transport stream if requested
            if output_file and len(ts_data) > 0:
                with open(output_file, 'wb') as f:
                    f.write(ts_data)
                dashboard.log("INFO", f"Saved TS to: {output_file}")
            
            # Log audio save (already saved by record() if from microphone)
            if save_audio and not input_file:
                dashboard.log("INFO", f"Saved recorded audio to: {save_audio}")
            
            # Save image if detected
            snap = stats.get_snapshot()
            if snap['image'].image_data:
                # Determine output path
                if output_image:
                    img_path = output_image
                else:
                    # Auto-generate filename
                    ext = snap['image'].image_format or 'jpg'
                    img_path = f"received_image.{ext}"
                
                with open(img_path, 'wb') as f:
                    f.write(snap['image'].image_data)
                dashboard.log("INFO", f"Saved image to: {img_path}")
            
            # Keep dashboard visible until Ctrl+C
            dashboard.log("INFO", "Processing complete! Press Ctrl+C to exit...")
            live.update(dashboard.get_renderable())
            
            # Keep refreshing until interrupted
            try:
                while True:
                    live.update(dashboard.get_renderable())
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        
        # Print summary after Live context ends
        if save_audio and not input_file:
            print(f"\n🎤 Audio saved: {save_audio}")
        
        snap = stats.get_snapshot()
        if snap['image'].image_data:
            img_path = output_image or f"received_image.{snap['image'].image_format or 'jpg'}"
            print(f"📷 Image saved: {img_path} ({snap['image'].width}x{snap['image'].height})")
        
        if result['error']:
            raise result['error']
        
        return ts_data, rx_stats
        
    finally:
        dashboard.stop()
        rx.close()

