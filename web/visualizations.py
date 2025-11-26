"""
Plotly Visualizations for DVB-T Pipeline Stages

Generates interactive Plotly figures for each stage of the DVB-T pipeline.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
from scipy import signal

# Handle both direct script execution and module import
try:
    from .pipeline import PipelineResults, DVBTParams
except ImportError:
    from pipeline import PipelineResults, DVBTParams


# Color scheme - dark theme with accent colors
COLORS = {
    'background': '#0d1117',
    'paper': '#161b22',
    'text': '#c9d1d9',
    'grid': '#30363d',
    'primary': '#58a6ff',
    'secondary': '#f78166',
    'tertiary': '#7ee787',
    'quaternary': '#d2a8ff',
    'accent1': '#ff7b72',
    'accent2': '#ffa657',
    'pilot_scattered': '#ff6b6b',
    'pilot_continual': '#4ecdc4',
    'pilot_tps': '#ffe66d',
    'data_carrier': '#58a6ff',
}


def _base_layout(title: str, height: int = 500) -> dict:
    """Create base layout for consistent styling."""
    return {
        'title': {
            'text': title,
            'font': {'size': 18, 'color': COLORS['text']},
        },
        'paper_bgcolor': COLORS['paper'],
        'plot_bgcolor': COLORS['background'],
        'font': {'color': COLORS['text']},
        'height': height,
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60},
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'zerolinecolor': COLORS['grid'],
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'zerolinecolor': COLORS['grid'],
        },
    }


def viz_input(results: PipelineResults) -> dict:
    """
    Visualize input transport stream analysis.
    
    Shows PID distribution and packet statistics.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('PID Distribution', 'Packet Statistics'),
        specs=[[{'type': 'bar'}, {'type': 'indicator'}]]
    )
    
    # PID distribution bar chart
    pids = list(results.pid_distribution.keys())
    counts = list(results.pid_distribution.values())
    
    # Create labels for known PIDs
    pid_labels = []
    for pid in pids:
        if pid == 0x0000:
            pid_labels.append('PAT (0x0000)')
        elif pid == 0x1FFF:
            pid_labels.append('NULL (0x1FFF)')
        elif pid < 0x0020:
            pid_labels.append(f'PSI (0x{pid:04X})')
        else:
            pid_labels.append(f'0x{pid:04X}')
    
    fig.add_trace(
        go.Bar(
            x=pid_labels,
            y=counts,
            marker_color=COLORS['primary'],
            text=counts,
            textposition='auto',
            name='Packets'
        ),
        row=1, col=1
    )
    
    # Packet count indicator
    fig.add_trace(
        go.Indicator(
            mode='number+delta',
            value=results.input_packets,
            title={'text': 'Total Packets'},
            number={'font': {'color': COLORS['primary'], 'size': 48}},
            delta={'reference': 0, 'relative': False},
        ),
        row=1, col=2
    )
    
    layout = _base_layout('Transport Stream Input Analysis', height=400)
    layout['showlegend'] = False
    fig.update_layout(**layout)
    
    return fig.to_dict()


def viz_scrambler(results: PipelineResults) -> dict:
    """
    Visualize energy dispersal (scrambling) effect.
    
    Shows bit distribution before and after scrambling.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Before Scrambling (Bit Distribution)',
            'After Scrambling (Bit Distribution)'
        )
    )
    
    # Before scrambling - histogram of running sum (shows patterns)
    if len(results.pre_scramble) > 0:
        # Calculate running average to show bit patterns
        window = min(100, len(results.pre_scramble) // 2)
        if window > 0:
            pre_avg = np.convolve(
                results.pre_scramble.astype(float), 
                np.ones(window)/window, 
                mode='valid'
            )
            
            fig.add_trace(
                go.Histogram(
                    x=pre_avg.tolist(),
                    nbinsx=50,
                    marker_color=COLORS['secondary'],
                    name='Before',
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # After scrambling
    if len(results.post_scramble) > 0:
        window = min(100, len(results.post_scramble) // 2)
        if window > 0:
            post_avg = np.convolve(
                results.post_scramble.astype(float),
                np.ones(window)/window,
                mode='valid'
            )
            
            fig.add_trace(
                go.Histogram(
                    x=post_avg.tolist(),
                    nbinsx=50,
                    marker_color=COLORS['tertiary'],
                    name='After',
                    opacity=0.8
                ),
                row=1, col=2
            )
    
    layout = _base_layout('Energy Dispersal (PRBS Scrambling)', height=400)
    layout['showlegend'] = True
    layout['annotations'] = [
        {
            'text': 'Scrambling flattens the bit distribution',
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.15,
            'showarrow': False,
            'font': {'size': 12, 'color': COLORS['text']},
        }
    ]
    fig.update_layout(**layout)
    
    fig.update_xaxes(title_text='Running Average (100 bits)', row=1, col=1)
    fig.update_xaxes(title_text='Running Average (100 bits)', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    
    return fig.to_dict()


def viz_reed_solomon(results: PipelineResults) -> dict:
    """
    Visualize Reed-Solomon encoding.
    
    Shows data bytes and parity bytes.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'RS(204,188) - First Codeword',
            'Parity Bytes Detail (16 bytes)'
        ),
        row_heights=[0.6, 0.4]
    )
    
    if len(results.rs_output) >= 204:
        # Full codeword visualization - convert to list for JSON
        x = list(range(204))
        colors = [COLORS['primary']] * 188 + [COLORS['accent1']] * 16
        
        fig.add_trace(
            go.Bar(
                x=x,
                y=results.rs_output[:204].tolist(),
                marker_color=colors,
                name='Codeword'
            ),
            row=1, col=1
        )
        
        # Add annotation to show data vs parity regions
        fig.add_vrect(
            x0=-0.5, x1=187.5,
            fillcolor=COLORS['primary'], opacity=0.1,
            line_width=0, row=1, col=1
        )
        fig.add_vrect(
            x0=187.5, x1=203.5,
            fillcolor=COLORS['accent1'], opacity=0.2,
            line_width=0, row=1, col=1
        )
    
    # Parity bytes detail - convert to list for JSON
    if len(results.rs_parity_example) >= 16:
        fig.add_trace(
            go.Bar(
                x=list(range(16)),
                y=results.rs_parity_example[:16].tolist(),
                marker_color=COLORS['accent1'],
                text=[f'0x{b:02X}' for b in results.rs_parity_example[:16]],
                textposition='outside',
                name='Parity'
            ),
            row=2, col=1
        )
    
    layout = _base_layout('Reed-Solomon RS(204,188) Encoding', height=550)
    layout['showlegend'] = False
    layout['annotations'] = [
        {
            'text': '<b>188 Data Bytes</b>',
            'x': 94, 'y': 280,
            'xref': 'x', 'yref': 'y',
            'showarrow': False,
            'font': {'size': 11, 'color': COLORS['primary']},
        },
        {
            'text': '<b>16 Parity</b>',
            'x': 196, 'y': 280,
            'xref': 'x', 'yref': 'y',
            'showarrow': False,
            'font': {'size': 11, 'color': COLORS['accent1']},
        }
    ]
    fig.update_layout(**layout)
    
    fig.update_xaxes(title_text='Byte Position', row=1, col=1)
    fig.update_xaxes(title_text='Parity Byte Index', row=2, col=1)
    fig.update_yaxes(title_text='Byte Value', row=1, col=1)
    fig.update_yaxes(title_text='Byte Value', row=2, col=1)
    
    return fig.to_dict()


def viz_convolutional(results: PipelineResults) -> dict:
    """
    Visualize convolutional encoding.
    
    Shows input bits and encoded output (rate 1/2 expansion).
    """
    n_input = min(500, len(results.conv_input))
    n_output = min(1000, len(results.conv_output))
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Input Bits ({n_input} shown)',
            f'Encoded Output (Rate 1/2 → {n_output} bits)'
        ),
        row_heights=[0.5, 0.5]
    )
    
    # Input bits - convert to list for JSON
    if n_input > 0:
        fig.add_trace(
            go.Scatter(
                x=list(range(n_input)),
                y=results.conv_input[:n_input].tolist(),
                mode='lines',
                line={'color': COLORS['primary'], 'width': 1},
                fill='tozeroy',
                fillcolor=f'rgba(88, 166, 255, 0.3)',
                name='Input'
            ),
            row=1, col=1
        )
    
    # Output bits - convert to list for JSON
    if n_output > 0:
        fig.add_trace(
            go.Scatter(
                x=list(range(n_output)),
                y=results.conv_output[:n_output].tolist(),
                mode='lines',
                line={'color': COLORS['tertiary'], 'width': 1},
                fill='tozeroy',
                fillcolor=f'rgba(126, 231, 135, 0.3)',
                name='Output'
            ),
            row=2, col=1
        )
    
    layout = _base_layout('Convolutional Encoding (K=7, Rate 1/2)', height=450)
    layout['showlegend'] = True
    layout['annotations'] = [
        {
            'text': 'Constraint length K=7, Generator polynomials G1=171₈, G2=133₈',
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.12,
            'showarrow': False,
            'font': {'size': 11, 'color': COLORS['text']},
        }
    ]
    fig.update_layout(**layout)
    
    fig.update_xaxes(title_text='Bit Index', row=1, col=1)
    fig.update_xaxes(title_text='Bit Index', row=2, col=1)
    fig.update_yaxes(title_text='Bit Value', range=[-0.1, 1.1], row=1, col=1)
    fig.update_yaxes(title_text='Bit Value', range=[-0.1, 1.1], row=2, col=1)
    
    return fig.to_dict()


def viz_puncturing(results: PipelineResults) -> dict:
    """
    Visualize puncturing pattern and effect.
    
    Shows the puncture pattern and how it reduces bit rate.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Puncture Pattern (Code Rate: {results.params.code_rate})',
            'Punctured Output Bits'
        ),
        row_heights=[0.3, 0.7]
    )
    
    # Puncture pattern visualization - convert to list for JSON
    pattern = results.puncture_pattern
    if len(pattern) > 0:
        # Repeat pattern for visualization
        n_repeat = max(1, 40 // len(pattern))
        full_pattern = np.tile(pattern, n_repeat)
        
        colors = [COLORS['tertiary'] if b else COLORS['accent1'] for b in full_pattern]
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(full_pattern))),
                y=full_pattern.tolist(),
                marker_color=colors,
                name='Pattern',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Punctured output - show whatever we have, convert to list for JSON
    n_show = min(1000, len(results.punctured))
    if n_show > 0:
        fig.add_trace(
            go.Scatter(
                x=list(range(n_show)),
                y=results.punctured[:n_show].tolist(),
                mode='lines',
                line={'color': COLORS['tertiary'], 'width': 1},
                fill='tozeroy',
                fillcolor=f'rgba(126, 231, 135, 0.3)',
                name='Punctured'
            ),
            row=2, col=1
        )
    
    # Calculate rate info
    rate_info = {
        '1/2': 'No puncturing',
        '2/3': '2 out of 4 bits kept',
        '3/4': '3 out of 6 bits kept',
        '5/6': '5 out of 10 bits kept',
        '7/8': '7 out of 14 bits kept',
    }
    
    layout = _base_layout('Rate Matching (Puncturing)', height=450)
    layout['showlegend'] = False
    layout['annotations'] = [
        {
            'text': f"{rate_info.get(results.params.code_rate, '')} — Green=Keep, Red=Puncture",
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.1,
            'showarrow': False,
            'font': {'size': 11, 'color': COLORS['text']},
        }
    ]
    fig.update_layout(**layout)
    
    fig.update_xaxes(title_text='Pattern Position', row=1, col=1)
    fig.update_xaxes(title_text='Bit Index', row=2, col=1)
    fig.update_yaxes(title_text='Keep/Puncture', range=[-0.1, 1.1], row=1, col=1)
    fig.update_yaxes(title_text='Bit Value', range=[-0.1, 1.1], row=2, col=1)
    
    return fig.to_dict()


def viz_qam(results: PipelineResults) -> dict:
    """
    Visualize QAM constellation diagram.
    
    Shows ideal constellation points and actual mapped symbols.
    """
    fig = go.Figure()
    
    constellation = results.params.constellation
    
    # Ideal constellation points - convert to list for proper JSON serialization
    if len(results.constellation_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=results.constellation_points.real.tolist(),
                y=results.constellation_points.imag.tolist(),
                mode='markers',
                marker={
                    'size': 15,
                    'color': COLORS['secondary'],
                    'symbol': 'diamond',
                    'line': {'width': 2, 'color': COLORS['text']}
                },
                name='Ideal Points'
            )
        )
    
    # Actual mapped symbols (subset for performance) - convert to list
    if len(results.qam_symbols) > 0:
        n_show = min(2000, len(results.qam_symbols))
        fig.add_trace(
            go.Scatter(
                x=results.qam_symbols[:n_show].real.tolist(),
                y=results.qam_symbols[:n_show].imag.tolist(),
                mode='markers',
                marker={
                    'size': 4,
                    'color': COLORS['primary'],
                    'opacity': 0.6
                },
                name='Mapped Symbols'
            )
        )
    
    # Add quadrant lines
    max_val = 1.5
    fig.add_hline(y=0, line_color=COLORS['grid'], line_width=1)
    fig.add_vline(x=0, line_color=COLORS['grid'], line_width=1)
    
    layout = _base_layout(f'{constellation} Constellation Diagram', height=550)
    layout['xaxis'] = {
        'title': 'In-Phase (I)',
        'range': [-max_val, max_val],
        'scaleanchor': 'y',
        'scaleratio': 1,
        'gridcolor': COLORS['grid'],
        'zerolinecolor': COLORS['grid'],
    }
    layout['yaxis'] = {
        'title': 'Quadrature (Q)',
        'range': [-max_val, max_val],
        'gridcolor': COLORS['grid'],
        'zerolinecolor': COLORS['grid'],
    }
    layout['showlegend'] = True
    layout['legend'] = {'x': 0.02, 'y': 0.98}
    
    bits_per_symbol = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[constellation]
    layout['annotations'] = [
        {
            'text': f'{constellation}: {bits_per_symbol} bits/symbol, {2**bits_per_symbol} constellation points',
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.1,
            'showarrow': False,
            'font': {'size': 11, 'color': COLORS['text']},
        }
    ]
    
    fig.update_layout(**layout)
    
    return fig.to_dict()


def viz_ofdm(results: PipelineResults) -> dict:
    """
    Visualize OFDM spectrum with pilots.
    
    Shows carrier amplitudes and pilot positions.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'OFDM Carrier Spectrum (One Symbol)',
            'Carrier Magnitude vs Frequency'
        ),
        row_heights=[0.5, 0.5]
    )
    
    if len(results.ofdm_carriers) > 0:
        carriers = results.ofdm_carriers
        n_carriers = len(carriers)
        x = np.arange(n_carriers).tolist()
        
        # Real and imaginary parts - convert to list for JSON
        fig.add_trace(
            go.Scatter(
                x=x,
                y=carriers.real.tolist(),
                mode='lines',
                line={'color': COLORS['primary'], 'width': 1},
                name='Real (I)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=carriers.imag.tolist(),
                mode='lines',
                line={'color': COLORS['quaternary'], 'width': 1},
                name='Imaginary (Q)'
            ),
            row=1, col=1
        )
        
        # Magnitude
        magnitude = np.abs(carriers)
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=magnitude.tolist(),
                mode='lines',
                line={'color': COLORS['tertiary'], 'width': 1},
                fill='tozeroy',
                fillcolor=f'rgba(126, 231, 135, 0.2)',
                name='Magnitude'
            ),
            row=2, col=1
        )
        
        # Mark pilot positions
        if len(results.pilot_positions) > 0:
            pilots = results.pilot_positions
            pilots = pilots[pilots < n_carriers]
            
            fig.add_trace(
                go.Scatter(
                    x=pilots.tolist(),
                    y=magnitude[pilots].tolist(),
                    mode='markers',
                    marker={
                        'size': 8,
                        'color': COLORS['pilot_scattered'],
                        'symbol': 'triangle-up'
                    },
                    name='Scattered Pilots'
                ),
                row=2, col=1
            )
    
    mode_info = {
        '2K': '2048-point FFT, 1705 active carriers',
        '8K': '8192-point FFT, 6817 active carriers',
    }
    
    layout = _base_layout(f'OFDM Spectrum ({results.params.mode} Mode)', height=500)
    layout['showlegend'] = True
    layout['legend'] = {'x': 0.02, 'y': 0.98}
    layout['annotations'] = [
        {
            'text': mode_info.get(results.params.mode, ''),
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.1,
            'showarrow': False,
            'font': {'size': 11, 'color': COLORS['text']},
        }
    ]
    fig.update_layout(**layout)
    
    fig.update_xaxes(title_text='Carrier Index', row=1, col=1)
    fig.update_xaxes(title_text='Carrier Index', row=2, col=1)
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Magnitude', row=2, col=1)
    
    return fig.to_dict()


def viz_output(results: PipelineResults) -> dict:
    """
    Visualize I/Q output waveform and spectrum.
    
    Shows time-domain samples and power spectral density.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'I/Q Time Domain (One OFDM Symbol)',
            'I/Q Scatter Plot',
            'Power Spectral Density',
            'Output Statistics'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'indicator'}]
        ]
    )
    
    if len(results.ofdm_symbol_example) > 0:
        symbol = results.ofdm_symbol_example
        t = (np.arange(len(symbol)) / results.sample_rate * 1e6).tolist()  # Convert to µs
        
        # Time domain I and Q - convert to list for JSON
        fig.add_trace(
            go.Scatter(
                x=t,
                y=symbol.real.tolist(),
                mode='lines',
                line={'color': COLORS['primary'], 'width': 1},
                name='I'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=t,
                y=symbol.imag.tolist(),
                mode='lines',
                line={'color': COLORS['quaternary'], 'width': 1},
                name='Q'
            ),
            row=1, col=1
        )
        
        # I/Q scatter plot - convert to list
        n_scatter = min(1000, len(symbol))
        fig.add_trace(
            go.Scatter(
                x=symbol[:n_scatter].real.tolist(),
                y=symbol[:n_scatter].imag.tolist(),
                mode='markers',
                marker={
                    'size': 2,
                    'color': COLORS['primary'],
                    'opacity': 0.5
                },
                name='I/Q',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Power Spectral Density
    if len(results.iq_samples) > 1024:
        # Use Welch's method for PSD
        f, psd = signal.welch(
            results.iq_samples[:min(65536, len(results.iq_samples))],
            fs=results.sample_rate,
            nperseg=1024,
            return_onesided=False
        )
        
        # Shift to center DC
        f = np.fft.fftshift(f) / 1e6  # Convert to MHz
        psd = np.fft.fftshift(psd)
        psd_db = 10 * np.log10(psd + 1e-12)
        
        fig.add_trace(
            go.Scatter(
                x=f.tolist(),
                y=psd_db.tolist(),
                mode='lines',
                line={'color': COLORS['tertiary'], 'width': 1},
                fill='tozeroy',
                fillcolor=f'rgba(126, 231, 135, 0.2)',
                name='PSD',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Statistics indicator
    fig.add_trace(
        go.Indicator(
            mode='number',
            value=results.data_rate / 1e6,
            title={'text': 'Data Rate (Mbps)'},
            number={
                'font': {'color': COLORS['primary'], 'size': 36},
                'suffix': '',
                'valueformat': '.2f'
            },
        ),
        row=2, col=2
    )
    
    layout = _base_layout('I/Q Output Waveform', height=550)
    layout['showlegend'] = True
    layout['legend'] = {'x': 0.02, 'y': 0.98}
    
    guard_info = {
        '1/4': '25%',
        '1/8': '12.5%',
        '1/16': '6.25%',
        '1/32': '3.125%',
    }
    
    layout['annotations'] = [
        {
            'text': f"Sample Rate: {results.sample_rate/1e6:.2f} MHz | Guard: {guard_info.get(results.params.guard_interval, '')} | Duration: {results.duration*1000:.1f} ms",
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.08,
            'showarrow': False,
            'font': {'size': 11, 'color': COLORS['text']},
        }
    ]
    fig.update_layout(**layout)
    
    fig.update_xaxes(title_text='Time (µs)', row=1, col=1)
    fig.update_xaxes(title_text='I', row=1, col=2)
    fig.update_xaxes(title_text='Frequency (MHz)', row=2, col=1)
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Q', row=1, col=2)
    fig.update_yaxes(title_text='Power (dB)', row=2, col=1)
    
    return fig.to_dict()


# Mapping of stage names to visualization functions
VISUALIZERS = {
    'input': viz_input,
    'scrambler': viz_scrambler,
    'reed_solomon': viz_reed_solomon,
    'convolutional': viz_convolutional,
    'puncturing': viz_puncturing,
    'qam': viz_qam,
    'ofdm': viz_ofdm,
    'output': viz_output,
}


def get_visualization(stage: str, results: PipelineResults) -> dict:
    """
    Get Plotly figure dict for a pipeline stage.
    
    Args:
        stage: Stage name (input, scrambler, reed_solomon, etc.)
        results: Pipeline results with intermediate data
        
    Returns:
        Plotly figure as dict (JSON-serializable)
    """
    visualizer = VISUALIZERS.get(stage)
    if visualizer is None:
        raise ValueError(f"Unknown stage: {stage}")
    
    return visualizer(results)


def get_all_visualizations(results: PipelineResults) -> Dict[str, dict]:
    """
    Get all Plotly figures for all stages.
    
    Args:
        results: Pipeline results
        
    Returns:
        Dict mapping stage name to Plotly figure dict
    """
    return {
        stage: visualizer(results)
        for stage, visualizer in VISUALIZERS.items()
    }

