"""Generate every figure of the paper into ./figures (vector PDF + 600-dpi PNG).
Run from the repository root:  python experiments/make_figures.py
Uses runs/extras.json (falls back to results/extras.json), runs/tsne.npz and runs/energy.npz."""
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 8, 'axes.grid': True, 'grid.alpha': .3, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'mathtext.fontset': 'dejavusans'})
OUT = 'figures'; os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/{name}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)

def box(ax, x, y, w, h, t, fc='white', fs=7, bold=False, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec='black', lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', linespacing=1.2)

def arr(ax, p, q, ls='-', c='black'):
    ax.annotate('', q, p, arrowprops=dict(arrowstyle='-|>', lw=0.8, ls=ls, color=c, mutation_scale=8))

# ---------------- Fig 1: system and threat model ----------------
fig, ax = plt.subplots(figsize=(174 * MM, 62 * MM)); ax.axis('off'); ax.set_xlim(0, 17.4); ax.set_ylim(0, 6.2)
ax.add_patch(Rectangle((0.05, 0.25), 4.75, 5.75, fc='#f4f4f4', ec='grey', lw=0.6, ls='--'))
ax.text(2.4, 5.65, 'IoT access network', ha='center', fontsize=7, weight='bold')
for i, y in enumerate([4.4, 3.3, 2.2]):
    box(ax, 0.2, y, 2.0, 0.85, f'Registered\ndevice {i + 1}, $\\Theta_{i + 1}$', fc='#dbe8f5', fs=6)
box(ax, 0.2, 0.5, 2.0, 1.2, 'Rogue or\ncloned device\nunknown $\\Theta$', fc='#f8d7d0', fs=6, ls='--')
box(ax, 2.65, 1.6, 2.0, 2.6, 'Wireless\nchannel\nmultipath,\nDoppler,\nfading, AWGN', fc='#e5f1df', fs=6)
for y in [4.82, 3.72, 2.62]: arr(ax, (2.2, y), (2.65, min(max(y, 1.9), 3.95)))
arr(ax, (2.2, 1.1), (2.65, 1.9), ls='--', c='#b03020')
ax.add_patch(FancyBboxPatch((5.15, 0.9), 7.0, 3.85, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#fff7e0', ec='black', lw=0.8))
ax.text(8.65, 4.35, 'Gateway or access point', ha='center', fontsize=7, weight='bold')
box(ax, 5.35, 2.45, 2.05, 1.4, 'RF front end\npreamble\ncapture, SNR $\\hat{s}$', fc='white', fs=6)
box(ax, 7.65, 2.45, 2.05, 1.4, 'SHIELD-RF\nencoder and\nprototype head', fc='#dbe8f5', fs=6)
box(ax, 9.95, 2.45, 2.0, 1.4, 'Energy test\n$E(\\mathbf{v})>\\tau_{b(\\hat{s})}$', fc='white', fs=6)
arr(ax, (4.65, 2.9), (5.35, 3.15)); arr(ax, (7.4, 3.15), (7.65, 3.15)); arr(ax, (9.7, 3.15), (9.95, 3.15))
box(ax, 5.35, 1.1, 6.6, 0.9, 'Prototype store: registered and enrolled devices', fc='white', fs=6)
arr(ax, (8.67, 2.0), (8.67, 2.45))
box(ax, 12.75, 3.45, 2.1, 1.0, 'ACCEPT\nidentity $k^{*}$', fc='#dbefd5', fs=6.3, bold=True)
box(ax, 12.75, 1.3, 2.1, 1.0, 'REJECT\nrogue alarm', fc='#f8d7d0', fs=6.3, bold=True)
arr(ax, (11.95, 3.4), (12.75, 3.95)); arr(ax, (11.95, 2.85), (12.75, 1.8))
ax.text(12.2, 3.85, 'no', fontsize=6, style='italic'); ax.text(12.2, 2.05, 'yes', fontsize=6, style='italic')
box(ax, 15.3, 2.2, 2.05, 1.6, 'Core\nnetwork and\nkey\nmanagement', fc='#eeeeee', fs=6)
arr(ax, (14.85, 3.95), (15.3, 3.4)); arr(ax, (14.85, 1.8), (15.3, 2.5), ls='--')
ax.text(16.3, 1.75, 'block and log', ha='center', fontsize=5.8, style='italic')
save(fig, 'Fig1')

# ---------------- Fig 2: architecture ----------------
fig, ax = plt.subplots(figsize=(174 * MM, 72 * MM)); ax.axis('off'); ax.set_xlim(0, 17.4); ax.set_ylim(0, 7.2)
ax.text(0.15, 6.75, '(a) Inference path', fontsize=7.5, weight='bold')
xs = [0.1, 3.1, 6.6, 9.9, 13.2]; ws = [2.7, 3.2, 3.0, 3.0, 4.1]
txt = ['Received burst $y[n]$\n$N=128$\npreamble samples', 'Dual view features $\\mathbf{v}$\nraw IQ, derotated IQ,\nlog spectrum (640)',
       'Encoder $f_\\theta$\nMLP 640, 256, 256,\n64 with $\\ell_2$ norm', 'Cosine prototype\nhead $\\ell_k = s\\,\\mathbf{z}^{T}\\hat{\\mathbf{p}}_k$\nenergy $E(\\mathbf{v})$',
       'SNR adaptive decision\nreject if $E(\\mathbf{v})>\\tau_{b(\\hat{s})}$\nelse identity $k^{*}$']
fcs = ['#e5f1df', '#dbe8f5', '#dbe8f5', '#dbe8f5', '#fff7e0']
for x, w, t, f in zip(xs, ws, txt, fcs): box(ax, x, 5.0, w, 1.45, t, fc=f, fs=6.2)
for i in range(4): arr(ax, (xs[i] + ws[i], 5.72), (xs[i + 1], 5.72))
ax.text(0.15, 4.3, '(b) Training pipeline', fontsize=7.5, weight='bold')
box(ax, 0.15, 2.35, 3.3, 1.6, 'Channel augmentation $\\mathcal{A}$\nrandom FIR, 1 to 4 taps,\nresidual Doppler,\nnoise 0 to 30 dB', fc='#e5f1df', fs=6.4)
box(ax, 3.85, 2.35, 3.3, 1.6, 'Stage 1\nsupervised contrastive\npretraining on two\nchannel views', fc='#eeeeee', fs=6.4)
box(ax, 7.55, 2.35, 3.3, 1.6, 'Stage 2\n$\\mathcal{L}_{CE}+\\lambda_p\\mathcal{L}_{proto}+\\lambda_e\\mathcal{L}_{E}$\nclean and augmented\nbursts', fc='#eeeeee', fs=6.4)
box(ax, 7.55, 0.2, 3.3, 1.5, 'IPOS, Algorithm 2\nvirtual rogue bursts by\nimpairment perturbation', fc='#f8d7d0', fs=6.4)
box(ax, 11.25, 2.35, 2.85, 1.6, 'Calibration\nper SNR band $b$:\n$\\tau_b$ = 95th percentile\nof legitimate energy', fc='#fff7e0', fs=6.4)
box(ax, 14.5, 2.35, 2.75, 1.6, 'Few shot enrollment\nnew prototype =\nnormalised mean of\n$K_s$ embeddings', fc='#fff7e0', fs=6.4)
arr(ax, (3.45, 3.15), (3.85, 3.15)); arr(ax, (7.15, 3.15), (7.55, 3.15)); arr(ax, (9.2, 1.7), (9.2, 2.35))
arr(ax, (10.85, 3.15), (11.25, 3.15)); arr(ax, (14.1, 3.15), (14.5, 3.15))
ax.plot([1.8, 1.8], [2.35, 0.95], color='black', lw=0.8); arr(ax, (1.8, 0.95), (7.55, 0.95))
ax.text(4.6, 1.1, 'fresh channel views of outliers', fontsize=6.2, style='italic', ha='center')
save(fig, 'Fig2')

# ---------------- Fig 3: IPOS block diagram ----------------
fig, ax = plt.subplots(figsize=(174 * MM, 42 * MM)); ax.axis('off'); ax.set_xlim(0, 17.4); ax.set_ylim(0, 4.0)
steps = [('Received\nburst $y$ of\ndevice $d$', '#dbe8f5'), ('IQ offset\n$\\tilde\\alpha$, $\\tilde\\varphi$\nwidely linear', '#f4f4f4'),
         ('PA offset\n$\\tilde a_3$, cubic\nnonlinearity', '#f4f4f4'), ('CFO offset\n$\\Delta\\tilde f$\nphase ramp', '#f4f4f4'),
         ('DC offset\n$\\tilde c$', '#f4f4f4'), ('Fresh channel\naugmentation\n$\\mathcal{A}$', '#e5f1df'),
         ('Virtual rogue\nburst $\\tilde y$\npushed to high\nenergy', '#f8d7d0')]
w, g = 2.1, 0.42
for i, (t, f) in enumerate(steps):
    x = 0.1 + i * (w + g); box(ax, x, 1.2, w, 1.7, t, fc=f, fs=6.5)
    if i: arr(ax, (x - g, 2.05), (x, 2.05))
x0 = 0.1 + (w + g) - 0.15; wd = 4 * w + 3 * g + 0.3
ax.add_patch(Rectangle((x0, 0.85), wd, 2.4, fc='none', ec='#b03020', lw=0.8, ls='--'))
ax.text(x0 + wd / 2, 3.45, 'Impairment perturbation $\\mathcal{T}_\\eta$ with random signs and magnitudes bounded away from zero', ha='center', fontsize=6.5, color='#b03020')
ax.text(x0 + wd / 2, 0.45, 'No capture of any rogue device is required', ha='center', fontsize=6.3, style='italic')
save(fig, 'Fig3')

# ---------------- data figures ----------------
src = 'runs/extras.json' if os.path.exists('runs/extras.json') else 'results/extras.json'
E = json.load(open(src))
COL = {'CE-DNN': '#7f7f7f', 'Center-Proto': '#1f77b4', 'DANN': '#2ca02c', 'SHIELD-RF': '#d62728'}
MK = {'CE-DNN': 'o', 'Center-Proto': 's', 'DANN': '^', 'SHIELD-RF': 'D'}
LS = {'CE-DNN': ':', 'Center-Proto': '--', 'DANN': '-.', 'SHIELD-RF': '-'}
s = E['snr']['snrs']
fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 60 * MM))
for a, key, yl, lab in [(axs[0], 'acc', 'Closed set accuracy', '(a)'), (axs[1], 'auroc', 'AUROC, registered vs rogue', '(b)')]:
    for m in COL: a.plot(s, E['snr'][m][key], marker=MK[m], color=COL[m], ls=LS[m], label=m, lw=1.2, ms=4)
    a.set_xlabel('SNR (dB)'); a.set_ylabel(yl); a.text(-0.17, 1.02, lab, transform=a.transAxes, fontsize=9, weight='bold')
axs[0].legend(fontsize=7)
save(fig, 'Fig4')

if os.path.exists('runs/tsne.npz'):
    T = np.load('runs/tsne.npz')
    fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 72 * MM))
    for a, k, lab in [(axs[0], 'ce', '(a) CE-DNN'), (axs[1], 'sh', '(b) SHIELD-RF')]:
        for d in range(10):
            for ci, mk in [(3, 'o'), (4, 'x')]:
                i = (T['y'] == d) & (T['c'] == ci)
                a.scatter(T[k][i, 0], T[k][i, 1], s=6, marker=mk, color=plt.cm.tab10(d), lw=0.6)
        a.set_xticks([]); a.set_yticks([]); a.text(0.0, 1.02, lab, transform=a.transAxes, fontsize=8, weight='bold')
    axs[1].scatter([], [], marker='o', color='k', s=8, label='Rayleigh'); axs[1].scatter([], [], marker='x', color='k', s=8, label='TDL-C')
    axs[1].legend(fontsize=7, loc='lower right'); save(fig, 'Fig5')

if os.path.exists('runs/energy.npz'):
    H = np.load('runs/energy.npz')
    fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 58 * MM))
    for a, m, lab in [(axs[0], 'CE-DNN', '(a) CE-DNN'), (axs[1], 'SHIELD-RF', '(b) SHIELD-RF')]:
        b = np.linspace(min(H[m + '_k'].min(), H[m + '_u'].min()), max(H[m + '_k'].max(), H[m + '_u'].max()), 45)
        a.hist(H[m + '_k'], b, density=True, histtype='stepfilled', alpha=.45, color='#1f77b4', label='registered devices')
        a.hist(H[m + '_u'], b, density=True, histtype='step', lw=1.2, hatch='///', color='#d62728', label='rogue devices')
        a.set_xlabel('Energy score $E(\\mathbf{v})$'); a.set_ylabel('Density'); a.text(0.0, 1.02, lab, transform=a.transAxes, fontsize=8, weight='bold')
    axs[0].legend(fontsize=7); save(fig, 'Fig6')

fig, ax = plt.subplots(figsize=(84 * MM, 62 * MM)); Ks = [1, 5, 10, 20]
for m in COL:
    f = E['fewshot'][m]
    ax.errorbar(Ks, [f[str(k)]['new'] for k in Ks], yerr=[f[str(k)]['new_sd'] for k in Ks], marker=MK[m], ls=LS[m],
                color=COL[m], label=m, capsize=2, lw=1.2, ms=4)
ax.set_xscale('log'); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.set_xlabel('Enrollment bursts per new device $K_s$')
ax.set_ylabel('Accuracy on enrolled devices'); ax.legend(fontsize=6.5); save(fig, 'Fig7')
print('figures written to', OUT)
