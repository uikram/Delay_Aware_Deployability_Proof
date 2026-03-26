"""
Hardware-in-the-Loop Simulation — Final Combined Version
=========================================================
Parameters derived from:
  [de Leva 1996]       M = 1.2 kg  (forearm+hand ~2.2% body mass, 54 kg subject)
  [Gallego et al 2010] fn = 3 Hz   (bounds voluntary motion < 2 Hz)
  [Ogata 2010]         zeta = 0.707 (Butterworth maximally-flat criterion)

  => K = M*(2*pi*fn)^2 = 426.3 N/m   (derived, not looked up)
  => B = 2*zeta*sqrt(K*M) = 32.0 N*s/m (derived)

Controller: PID with gains INDEPENDENT of plant K/B.
  Kp=1200, Ki=1000, Kd=5  (Kp >> K ensures low steady-state error)
  Validated response:
    tau=10ms  -> OS= 5.6%,  settling ~2.6s  (stable)
    tau=15ms  -> OS=14.0%,  settling ~2.6s  (stable)
    tau=20ms  -> OS=23.3%,  settling ~2.6s  (stable)
    tau=30ms  -> OS=43.5%,  settling ~1.2s  (marginal)
    tau=50ms  -> OS=970%,   UNSTABLE        (exceeds tau_max ~ 38ms)

Metrics (Gemini-compatible naming):
  IAE = Integral of Absolute Error
  ISE = Integral of Squared Error
  OS% = peak overshoot
  Ts  = settling time into +-5% band (sustained 50ms)
        [5% band used because delayed PID systems have persistent low-amplitude
         oscillation; 5% is standard for underdamped robotic interfaces]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, os
import warnings; warnings.filterwarnings('ignore')

# Set output directory
OUTPUT_DIR = '../results_attained'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. DERIVED PLANT PARAMETERS ───────────────────────────────────────────────
M       = 1.2
f_n     = 3.0
zeta    = 0.707
omega_n = 2 * np.pi * f_n           # 18.85 rad/s
K       = M * omega_n**2             # 426.3 N/m
B       = 2 * zeta * np.sqrt(K * M)  # 32.0  N·s/m

print("=" * 58)
print("DERIVED PLANT PARAMETERS")
print(f"  M        = {M} kg          [de Leva 1996]")
print(f"  f_n      = {f_n} Hz         [Gallego et al. 2010]")
print(f"  zeta     = {zeta}          [Ogata 2010, Butterworth]")
print(f"  omega_n  = {omega_n:.4f} rad/s")
print(f"  K        = {K:.2f} N/m")
print(f"  B        = {B:.2f} N·s/m")
print("=" * 58)

# ── 2. PID CONTROLLER GAINS (independent of plant K/B) ───────────────────────
Kp = 1200.0
Ki = 1000.0
Kd =    5.0

# ── 3. SIMULATION SETTINGS ───────────────────────────────────────────────────
dt      = 0.0005            # 0.5 ms  —  2 kHz (matches embedded controllers)
T_end   = 3.0               # seconds
t_span  = np.arange(0, T_end, dt)
N       = len(t_span)
x_ref   = 1.0               # unit step [m or rad]

delay_ms_list = [10, 15, 20, 30, 50]

# Measured latency values from paper
lora_m_ms   = 13.62   
lora_um_ms  = 21.65 
tau_max_ms  = 38.0   # Lyapunov-Krasovskii / Pade stability bound

# ── 4. SIMULATION FUNCTION ────────────────────────────────────────────────────
def simulate(tau_ms):
    """PID-controlled MSD with perception delay tau_ms [ms]."""
    delay_steps = int((tau_ms / 1000.0) / dt)
    x        = np.zeros(N)
    x_dot    = np.zeros(N)
    integral = 0.0

    for i in range(1, N):
        i_d    = max(0, i - delay_steps)
        e      =  x_ref - x[i_d]
        e_dot  = -x_dot[i_d]
        integral += e * dt

        u      = Kp * e + Ki * integral + Kd * e_dot
        x_ddot = (u - B * x_dot[i-1] - K * x[i-1]) / M

        # Semi-implicit Euler integration
        x_dot[i] = x_dot[i-1] + x_ddot * dt
        x[i]     = x[i-1]     + x_dot[i] * dt
        x[i]     = np.clip(x[i], -6, 12)   # clip for display only

    return t_span, x, x_dot

# ── 5. METRIC EXTRACTION ─────────────────────────────────────────────────────
def compute_metrics(t, x):
    error = x_ref - x

    iae = np.sum(np.abs(error)) * dt             # Tracking Error (IAE)
    ise = np.sum(error**2) * dt                  # Oscillation Energy (ISE)
    overshoot = max(0.0, (np.max(x) - x_ref) / x_ref * 100.0)

    # Settling time: +-5% band, sustained 50 ms
    band    = 0.05 * x_ref
    sustain = int(0.05 / dt)
    ts      = np.nan
    for i in range(N - sustain):
        if np.all(np.abs(x[i: i + sustain] - x_ref) <= band):
            ts = t[i]
            break

    return dict(iae=iae, ise=ise, overshoot=overshoot, settling_time=ts)

# ── 6. RUN ALL DELAYS ─────────────────────────────────────────────────────────
print(f"\n{'tau(ms)':<10} {'IAE':<10} {'OS(%)':<10} {'ISE':<12} {'Ts(s)'}")
print("-" * 56)

results      = {}
metrics_dict = {}

for tau_ms in delay_ms_list:
    t, x, xd          = simulate(tau_ms)
    m                 = compute_metrics(t, x)
    results[tau_ms]   = (t, x)
    metrics_dict[tau_ms] = m
    ts_str = f"{m['settling_time']:.3f}" if not np.isnan(m['settling_time']) else "Unstable"
    print(f"{tau_ms:<10} {m['iae']:<10.4f} {m['overshoot']:<10.2f} {m['ise']:<12.5f} {ts_str}")

# ── 7. JSON EXPORT (Gemini-compatible format) ─────────────────────────────────
json_out = {}
for tau_ms in delay_ms_list:
    m  = metrics_dict[tau_ms]
    ts = f"{m['settling_time']:.4f}" if not np.isnan(m['settling_time']) else "Unstable"
    json_out[f"{tau_ms}ms"] = {
        "Tracking Error (IAE)":     round(m['iae'],       4),
        "Overshoot (%)":            round(m['overshoot'], 2),
        "Oscillation Energy (ISE)": round(m['ise'],       4),
        "Settling Time (s)":        ts
    }
with open(os.path.join(OUTPUT_DIR, 'hil_delay_metrics.json'), 'w') as f:
    json.dump(json_out, f, indent=4)
print(f"\n✓ Saved -> {OUTPUT_DIR}/hil_delay_metrics.json")

# ── 8. PLOT HELPERS ───────────────────────────────────────────────────────────
COLORS = ['#27ae60', '#2980b9', '#e67e22', '#d35400', '#c0392b']
BG     = '#ffffff'
PANEL  = '#ffffff'
GRID_C = '#e0e0e0'
TEXT_C = '#000000'

# Distinct colors for LoRA/tau_max markers — never reused in step response lines
COLOR_LORA_M  = '#8e44ad'   # deep purple  — LoRA merged
COLOR_LORA_UM = '#16a085'   # teal         — LoRA unmerged
COLOR_TAU_MAX = '#c0392b'   # dark red     — stability boundary

def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color('#333333')
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.grid(True, color=GRID_C, lw=0.6, zorder=0)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_C, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_C, fontsize=10)
    if title:  ax.set_title(title, color=TEXT_C, fontsize=11, fontweight='bold')

def add_model_lines(ax):
    ax.axvspan(0,          tau_max_ms, alpha=0.08, color='#27ae60', zorder=1)
    ax.axvspan(tau_max_ms, 55,         alpha=0.08, color='#e74c3c', zorder=1)
    ax.axvline(lora_m_ms,  color=COLOR_LORA_M,  lw=1.6, ls='--',
               label=f'LoRA merged  ({lora_m_ms} ms)')
    ax.axvline(lora_um_ms, color=COLOR_LORA_UM, lw=1.6, ls='--',
               label=f'LoRA unmerged ({lora_um_ms} ms)')
    ax.axvline(tau_max_ms, color=COLOR_TAU_MAX, lw=1.8, ls='-',
               label=f'$\\tau_{{max}}$ = {tau_max_ms} ms')
    ax.text(0.13, 0.91, 'STABLE',   transform=ax.transAxes,
            color='#27ae60', fontsize=8.5, fontweight='bold')
    ax.text(0.76, 0.91, 'DEGRADED', transform=ax.transAxes,
            color='#c0392b', fontsize=8.5, fontweight='bold')

# ── FIGURE 1: Step Responses ──────────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig1.patch.set_facecolor(BG)

for ax, xlim, title in [
    (ax1, T_end, 'Closed-Loop Step Response Under Perception Delay'),
    (ax2, 0.8,   'Transient Detail  (0 - 0.8 s)')
]:
    for tau_ms, c in zip(delay_ms_list, COLORS):
        t, x = results[tau_ms]
        ls   = ':' if tau_ms == 50 else '-'
        lbl  = f'tau = {tau_ms} ms' + (' [UNSTABLE]' if tau_ms == 50 else '')
        ax.plot(t, x, color=c, lw=1.9, ls=ls, label=lbl, zorder=3)

    ax.axhline(x_ref,      color='black', lw=1.3, ls='--', alpha=0.7, label='Reference')
    ax.axhline(x_ref*1.05, color='gray',  lw=0.6, ls=':',  alpha=0.4)
    ax.axhline(x_ref*0.95, color='gray',  lw=0.6, ls=':',  alpha=0.4)
    ax.set_xlim(0, xlim)
    ax.set_ylim(-0.1, 3.0 if xlim == T_end else 1.9)
    ax.legend(fontsize=8.5, facecolor='white', edgecolor='#333',
              labelcolor='black', framealpha=1.0, loc='upper right')
    style_ax(ax, xlabel='Time (s)', ylabel='Position (rad)', title=title)

plt.tight_layout(pad=2.2)
fig1_path = os.path.join(OUTPUT_DIR, 'fig1_step_response')
plt.savefig(f'{fig1_path}.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.savefig(f'{fig1_path}.pdf', format='pdf', bbox_inches='tight', facecolor=BG)
plt.close()
print(f"✓ Saved {fig1_path}.png & .pdf")

# ── FIGURE 2: Four-panel degradation curves ───────────────────────────────────
tau_x = delay_ms_list
iae_y = [metrics_dict[m]['iae']       for m in delay_ms_list]
os_y  = [metrics_dict[m]['overshoot'] for m in delay_ms_list]
ise_y = [metrics_dict[m]['ise']       for m in delay_ms_list]
st_y  = [metrics_dict[m]['settling_time']
         if not np.isnan(metrics_dict[m]['settling_time']) else T_end
         for m in delay_ms_list]

panels = [
    ('Tracking Error - IAE (m*s)',       '#3498db', iae_y, False),
    ('Overshoot (%)',                     '#e74c3c', os_y,  True),
    ('Oscillation Energy - ISE (m^2*s)', '#9b59b6', ise_y, True),
    ('Settling Time (s)',                 '#f39c12', st_y,  False),
]

fig2 = plt.figure(figsize=(14, 9))
fig2.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.46, wspace=0.34)

for idx, (ylabel, color, ydata, use_log) in enumerate(panels):
    ax = fig2.add_subplot(gs[idx//2, idx%2])
    ax.plot(tau_x, ydata, color=color, lw=2.2,
            marker='o', markersize=8,
            markerfacecolor='white', markeredgecolor=color, zorder=4)
    if use_log and min(ydata) > 0:
        ax.set_yscale('log')
    add_model_lines(ax)
    ax.set_xticks(tau_x)
    ax.legend(fontsize=7.8, facecolor='white', edgecolor='#333',
              labelcolor='black', framealpha=1.0, loc='upper left')
    style_ax(ax, xlabel='Perception Delay (ms)', ylabel=ylabel)

fig2.suptitle(
    'Latency-Stability Degradation Curves\n'
    f'PID-Controlled MSD Plant  M={M} kg  K={K:.1f} N/m  B={B:.1f} N*s/m  '
    f'($\\omega_n$={omega_n:.2f} rad/s  $\\zeta$={zeta})',
    color=TEXT_C, fontsize=12, fontweight='bold', y=0.998)

fig2_path = os.path.join(OUTPUT_DIR, 'fig2_degradation_curves')
plt.savefig(f'{fig2_path}.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.savefig(f'{fig2_path}.pdf', format='pdf', bbox_inches='tight', facecolor=BG)
plt.close()
print(f"✓ Saved {fig2_path}.png & .pdf")

# ── FIGURE 3: Dual-axis degradation (Gemini-style layout) ─────────────────────
fig3, ax_main = plt.subplots(figsize=(9, 5.5))
fig3.patch.set_facecolor(BG)
ax_main.set_facecolor(PANEL)
for sp in ax_main.spines.values(): sp.set_color('#333333')
ax_twin = ax_main.twinx()

l1, = ax_main.plot(tau_x, os_y,  'o-',  color='#c0392b', lw=2.5,
                    markersize=8, label='Overshoot (%)')
l2, = ax_twin.plot( tau_x, ise_y, 's--', color='#2980b9', lw=2.5,
                    markersize=8, label='Osc. Energy (ISE)')

ax_main.set_yscale('log')
ax_twin.set_yscale('log')

# Shade zones and model lines (on main axis)
ax_main.axvspan(0,          tau_max_ms, alpha=0.08, color='#27ae60', zorder=1)
ax_main.axvspan(tau_max_ms, 55,         alpha=0.08, color='#e74c3c', zorder=1)
ax_main.axvline(lora_m_ms,  color=COLOR_LORA_M,  lw=1.5, ls='--')
ax_main.axvline(lora_um_ms, color=COLOR_LORA_UM, lw=1.5, ls='--')
ax_main.axvline(tau_max_ms, color=COLOR_TAU_MAX, lw=1.5, ls='-')
ax_main.text(0.13, 0.91, 'STABLE',   transform=ax_main.transAxes,
             color='#27ae60', fontsize=9, fontweight='bold')
ax_main.text(0.76, 0.91, 'DEGRADED', transform=ax_main.transAxes,
             color='#c0392b', fontsize=9, fontweight='bold')

ax_main.set_xlabel('Perception Delay (ms)', color=TEXT_C, fontsize=11)
ax_main.set_ylabel('Overshoot (%) — Log Scale',          color='#c0392b', fontsize=10, fontweight='bold')
ax_twin.set_ylabel('Oscillation Energy ISE — Log Scale', color='#2980b9', fontsize=10, fontweight='bold')
ax_main.tick_params(colors=TEXT_C); ax_twin.tick_params(colors=TEXT_C)
ax_main.set_xticks(tau_x)
ax_main.grid(True, which='both', color=GRID_C, lw=0.5)
ax_main.set_title('Latency-Stability Degradation Curves (Dual Axis)',
                   color=TEXT_C, fontsize=12, fontweight='bold')

extra_lines = [
    plt.Line2D([0],[0], color=COLOR_LORA_M,  lw=1.5, ls='--', label=f'LoRA merged ({lora_m_ms} ms)'),
    plt.Line2D([0],[0], color=COLOR_LORA_UM, lw=1.5, ls='--', label=f'LoRA unmerged ({lora_um_ms} ms)'),
    plt.Line2D([0],[0], color=COLOR_TAU_MAX, lw=1.5, ls='-',  label=f'$\\tau_{{max}}$={tau_max_ms} ms'),
]
ax_main.legend(handles=[l1, l2] + extra_lines, fontsize=8.5,
               facecolor='white', edgecolor='#333', labelcolor='black', framealpha=1.0)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'fig3_dual_axis')
plt.savefig(f'{fig3_path}.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.savefig(f'{fig3_path}.pdf', format='pdf', bbox_inches='tight', facecolor=BG)
plt.close()
print(f"✓ Saved {fig3_path}.png & .pdf")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL METRICS TABLE FOR PAPER")
print("=" * 70)
print(f"{'tau(ms)':<10} {'IAE':<12} {'OS(%)':<12} {'ISE':<16} {'Ts(5% band)'}")
print("-" * 70)
for tau_ms in delay_ms_list:
    m  = metrics_dict[tau_ms]
    ts = f"{m['settling_time']:.3f}" if not np.isnan(m['settling_time']) else "Unstable"
    print(f"{tau_ms:<10} {m['iae']:<12.4f} {m['overshoot']:<12.2f} {m['ise']:<16.5f} {ts}")

print(f"\nPlant      : M={M}kg  K={K:.2f}N/m  B={B:.2f}N*s/m")
print(f"             wn={omega_n:.4f}rad/s  zeta={zeta}  fn={f_n}Hz")
print(f"Controller : Kp={Kp}  Ki={Ki}  Kd={Kd}  (independent of plant K/B)")
print(f"tau_max    : {tau_max_ms} ms  (Lyapunov-Krasovskii / Pade analysis)")
print(f"\nDeployability map:")
print(f"  LoRA-CLIP merged   {lora_m_ms}ms  -> {lora_m_ms/tau_max_ms*100:.1f}% of tau_max  TIER 1 SAFE")
print(f"  LoRA-CLIP unmerged {lora_um_ms}ms  -> {lora_um_ms/tau_max_ms*100:.1f}% of tau_max  TIER 2 SAFE")
print(f"  Frozen Prefix-LM   396.67ms  -> {396.67/tau_max_ms:.1f}x tau_max     UNSTABLE")