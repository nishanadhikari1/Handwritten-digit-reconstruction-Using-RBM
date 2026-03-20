"""
RBM Digit Generator — GENERATE GUI
Uses 10 separate RBMs (one per digit).
Selecting digit 7 loads RBM-7 and samples from it.

Run AFTER training:  python rbm_generate.py
"""

import torch
import matplotlib
matplotlib.use('TkAgg')       
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
import gc, os

torch.set_num_threads(6)


# ──────────────────────────────────────────────────
#  RBM (same as training)
# ──────────────────────────────────────────────────
class RBM:
    def __init__(self, n_vis=784, n_hid=256):
        self.W  = torch.zeros(n_hid, n_vis)
        self.bv = torch.zeros(n_vis)
        self.bh = torch.zeros(n_hid)

    def h_given_v(self, v):
        p = torch.sigmoid(v @ self.W.T + self.bh)
        return p, torch.bernoulli(p)

    def v_given_h(self, h):
        p = torch.sigmoid(h @ self.W + self.bv)
        return p, torch.bernoulli(p)

    @torch.no_grad()
    def sample(self, n=9, steps=2000):
        v = torch.bernoulli(torch.rand(n, self.bv.size(0)) * 0.1)
        for _ in range(steps):
            _, h  = self.h_given_v(v)
            pv, v = self.v_given_h(h)
        _, h  = self.h_given_v(v)
        pv, _ = self.v_given_h(h)
        return pv

    @torch.no_grad()
    def reconstruct(self, v_noisy, steps=20):
        v = v_noisy.clone()
        for _ in range(steps):
            _, h  = self.h_given_v(v)
            pv, v = self.v_given_h(h)
        _, h  = self.h_given_v(v)
        pv, _ = self.v_given_h(h)
        return pv

    def load_state_dict(self, d):
        self.W  = d['W']
        self.bv = d['bv']
        self.bh = d['bh']


# ──────────────────────────────────────────────────
#  Load all 10 RBMs
# ──────────────────────────────────────────────────
def load_rbms():
    path = 'rbm_model.pth'
    if not os.path.exists(path):
        print("\n✗  rbm_model.pth not found.")
        print("   Run  python rbm_train.py  first.\n")
        exit(1)

    ck = torch.load(path, map_location='cpu')
    if 'rbms' not in ck:
        print("\n✗  Incompatible model. Delete rbm_model.pth and retrain.\n")
        exit(1)

    rbms = {}
    for d in range(10):
        r = RBM(n_vis=ck['n_vis'], n_hid=ck['n_hid'])
        r.load_state_dict(ck['rbms'][d])
        rbms[d] = r

    print(f"✓ Loaded 10 RBMs  (n_hid={ck['n_hid']})")
    return rbms


# ──────────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────────
def launch_gui(rbms):
    BG    = '#0d0d14'
    PANEL = '#13131e'
    ACC   = '#6d28d9'
    ACC2  = '#0891b2'
    TEXT  = '#e2e8f0'
    MUTED = '#475569'
    GRN   = '#22c55e'
    YLW   = '#fbbf24'

    state = {'digit': 0, 'sel': 0}
    buf   = [None]

    def do_gen(digit, steps):
        print(f"  Sampling 9 × digit-{digit} "
              f"from RBM-{digit} ({steps} steps)…",
              end='', flush=True)
        s = rbms[digit].sample(n=9, steps=steps)
        gc.collect()
        print(" done.")
        return s

    print("\n  Initial generation (digit 0)…")
    buf[0] = do_gen(0, 2000)

    # ── figure ──────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title('RBM Digit Generator')
    except Exception:
        pass

    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.015, right=0.985, top=0.915, bottom=0.02,
        wspace=0.05, width_ratios=[2.3, 1])

    # 3×3 grid
    g_grid = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=outer[0], hspace=0.04, wspace=0.04)
    axg = [fig.add_subplot(g_grid[r, c])
           for r in range(3) for c in range(3)]

    # right controls
    g_ctrl = gridspec.GridSpecFromSubplotSpec(
        9, 1, subplot_spec=outer[1], hspace=0.38,
        height_ratios=[2.2, 0.38, 0.38, 0.48,
                       0.44, 0.48, 0.44, 0.44, 0.44])
    ax_det  = fig.add_subplot(g_ctrl[0])
    ax_st   = fig.add_subplot(g_ctrl[1])
    ax_dlbl = fig.add_subplot(g_ctrl[2])
    ax_dbtn = fig.add_subplot(g_ctrl[3])
    ax_bgen = fig.add_subplot(g_ctrl[4])
    ax_gsl  = fig.add_subplot(g_ctrl[5])
    ax_bsv  = fig.add_subplot(g_ctrl[6])
    ax_nsl  = fig.add_subplot(g_ctrl[7])
    ax_bdn  = fig.add_subplot(g_ctrl[8])

    fig.text(0.5, 0.966,
             'RBM Handwritten Digit Generator',
             ha='center', color=TEXT, fontsize=14,
             fontweight='bold', fontfamily='monospace')
    fig.text(0.5, 0.946,
             '10 RBMs  ·  one per digit  ·  CD-1  ·  Gibbs sampling',
             ha='center', color=MUTED, fontsize=9,
             fontfamily='monospace')

    # ── draw helpers ────────────────────────────────
    def draw_grid(s):
        for i, ax in enumerate(axg):
            ax.clear()
            ax.set_facecolor('black')
            ax.imshow(s[i].view(28, 28).numpy(),
                      cmap='gray', interpolation='bilinear',
                      aspect='equal')
            ax.axis('off')
            for sp in ax.spines.values():
                sp.set_edgecolor(ACC)
                sp.set_linewidth(0.8)

    def draw_detail(idx, title=None):
        state['sel'] = idx
        ax_det.clear()
        ax_det.set_facecolor('black')
        ax_det.imshow(buf[0][idx].view(28, 28).numpy(),
                      cmap='gray', interpolation='bilinear',
                      aspect='equal')
        t = title or f"Digit  {state['digit']}  —  sample #{idx+1}"
        ax_det.set_title(t, color=TEXT, fontsize=10,
                         fontfamily='monospace', pad=4)
        ax_det.axis('off')
        for sp in ax_det.spines.values():
            sp.set_edgecolor(ACC2)
            sp.set_linewidth(1.4)

    def set_status(msg, color=GRN):
        ax_st.clear()
        ax_st.axis('off')
        ax_st.set_facecolor(PANEL)
        ax_st.text(0.5, 0.5, msg,
                   ha='center', va='center',
                   color=color, fontsize=9,
                   fontfamily='monospace',
                   transform=ax_st.transAxes)

    def on_click(ev):
        for i, ax in enumerate(axg):
            if ev.inaxes is ax:
                draw_detail(i)
                fig.canvas.draw_idle()
                break

    fig.canvas.mpl_connect('button_press_event', on_click)

    # ── "Select digit:" label ────────────────────────
    ax_dlbl.axis('off')
    ax_dlbl.set_facecolor(PANEL)
    ax_dlbl.text(0.5, 0.5, 'Select digit to generate:',
                 ha='center', va='center',
                 color=MUTED, fontsize=9,
                 fontfamily='monospace',
                 transform=ax_dlbl.transAxes)

    # ── 10 digit buttons ─────────────────────────────
    g_db = gridspec.GridSpecFromSubplotSpec(
        1, 10, subplot_spec=ax_dbtn.get_subplotspec(), wspace=0.06)
    ax_dbtn.remove()

    dax  = []
    dbtn = []
    for d in range(10):
        a = fig.add_subplot(g_db[0, d])
        a.set_facecolor(PANEL)
        b = Button(a, str(d),
                   color='#1e1b4b', hovercolor='#3730a3')
        b.label.set_color(TEXT)
        b.label.set_fontfamily('monospace')
        b.label.set_fontsize(12)
        b.label.set_fontweight('bold')
        dax.append(a)
        dbtn.append(b)

    def hl(d):
        for i in range(10):
            c = '#4c1d95' if i == d else '#1e1b4b'
            dax[i].set_facecolor(c)
            dbtn[i].ax.set_facecolor(c)
        fig.canvas.draw_idle()

    def mk(d):
        def h(ev):
            state['digit'] = d
            hl(d)
            set_status(
                f'Digit  {d}  selected — click  Generate ⟳', ACC2)
            fig.canvas.draw_idle()
        return h

    for d in range(10):
        dbtn[d].on_clicked(mk(d))

    hl(0)

    # ── generate button ──────────────────────────────
    btn_gen = Button(ax_bgen, '⟳   Generate Digit',
                     color='#1e1b4b', hovercolor='#3730a3')
    btn_gen.label.set_color(TEXT)
    btn_gen.label.set_fontfamily('monospace')
    btn_gen.label.set_fontsize(11)

    # ── gibbs slider ─────────────────────────────────
    ax_gsl.set_facecolor(PANEL)
    sl_g = Slider(ax_gsl, 'Gibbs', 500, 4000,
                  valinit=2000, valstep=500, color=ACC)
    sl_g.label.set_color(TEXT)
    sl_g.valtext.set_color(ACC2)

    def on_gen(ev):
        d     = state['digit']
        steps = int(sl_g.val)
        set_status(f'⏳  Sampling from RBM-{d}…  ({steps} steps)', YLW)
        fig.canvas.draw()
        fig.canvas.flush_events()
        buf[0] = do_gen(d, steps)
        draw_grid(buf[0])
        draw_detail(state['sel'])
        set_status(
            f'✓  9 samples of digit  {d}  —  {steps} Gibbs steps', GRN)
        fig.canvas.draw_idle()

    btn_gen.on_clicked(on_gen)

    # ── save button ───────────────────────────────────
    btn_sv = Button(ax_bsv, '⤓   Save Grid PNG',
                    color='#14532d', hovercolor='#166534')
    btn_sv.label.set_color(TEXT)
    btn_sv.label.set_fontfamily('monospace')
    btn_sv.label.set_fontsize(10)

    def on_save(ev):
        d    = state['digit']
        path = f'generated_digit_{d}.png'
        fig2, ax2 = plt.subplots(3, 3, figsize=(5, 5),
                                 facecolor='black')
        for i, a in enumerate(ax2.flat):
            a.imshow(buf[0][i].view(28, 28).numpy(),
                     cmap='gray', interpolation='bilinear')
            a.axis('off')
            a.set_facecolor('black')
        plt.suptitle(f'RBM  Generated  Digit  {d}',
                     color='white', fontsize=11)
        plt.tight_layout()
        plt.savefig(path, dpi=120, facecolor='black')
        plt.close(fig2)
        gc.collect()
        set_status(f'✓  Saved  →  {path}', GRN)
        fig.canvas.draw_idle()
        print(f"  ✓ Saved {path}")

    btn_sv.on_clicked(on_save)

    # ── noise slider ──────────────────────────────────
    ax_nsl.set_facecolor(PANEL)
    sl_n = Slider(ax_nsl, 'Noise', 0.1, 0.9,
                  valinit=0.4, valstep=0.1, color='#0891b2')
    sl_n.label.set_color(TEXT)
    sl_n.valtext.set_color(ACC2)

    # ── denoise button ────────────────────────────────
    btn_dn = Button(ax_bdn, '⟲   Denoise Selected',
                    color='#1e3a5f', hovercolor='#1e4d7b')
    btn_dn.label.set_color(TEXT)
    btn_dn.label.set_fontfamily('monospace')
    btn_dn.label.set_fontsize(10)

    def on_dn(ev):
        idx   = state['sel']
        d     = state['digit']
        noise = sl_n.val
        clean = buf[0][idx]
        noisy = torch.bernoulli(
            (clean + noise * torch.rand_like(clean)).clamp(0, 1))
        recon = rbms[d].reconstruct(
            noisy.unsqueeze(0), steps=30).squeeze(0)

        ax_det.clear()
        ax_det.set_facecolor('black')
        combined = torch.cat([noisy, recon]).view(28, 56).numpy()
        ax_det.imshow(combined, cmap='gray',
                      interpolation='bilinear', aspect='equal')
        ax_det.axvline(x=27.5, color=ACC2, lw=1.5, alpha=0.9)
        ax_det.set_title(
            f'Digit {d}  —  noisy  |  denoised',
            color=TEXT, fontsize=9, fontfamily='monospace')
        ax_det.axis('off')
        set_status(
            f'✓  Digit {d} denoised  (noise={noise:.1f})', GRN)
        fig.canvas.draw_idle()
        del noisy, recon, combined
        gc.collect()

    btn_dn.on_clicked(on_dn)

    # style all control axes
    for ax in [ax_st, ax_dlbl, ax_bgen, ax_gsl,
               ax_bsv, ax_nsl, ax_bdn]:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor('#1e293b')

    # ── initial render ───────────────────────────────
    draw_grid(buf[0])
    draw_detail(0)
    set_status('✓  Ready — pick a digit and click  Generate ⟳', GRN)

    plt.show()
    gc.collect()


# ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n╔══════════════════════════════════════════╗")
    print("║  RBM Digit Generator — Generate GUI      ║")
    print("╚══════════════════════════════════════════╝\n")
    rbms = load_rbms()
    print("\nHow to use:")
    print("  1. Click a digit button [0]–[9]")
    print("  2. Click  [Generate Digit]")
    print("  3. All 9 grid cells = samples of that digit")
    print("  4. Click any cell to zoom in\n")
    launch_gui(rbms)