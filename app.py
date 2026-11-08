import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------- 1. CONSTANTS ----------
K_B = 8.617e-5          # Boltzmann constant (eV/K)
Q = 1.602e-19           # electron charge (C)
EPS_SI = 11.7 * 8.854e-14   # permittivity of silicon (F/cm)
EG = 1.12               # silicon band gap (eV)
NI_300 = 1.5e10         # intrinsic carrier conc. at 300 K (cm^-3)
AREA = 1e-4             # junction area (cm^2)
DN, DP = 25.0, 10.0     # diffusion coefficients (cm^2/s)
TAU = 1e-6              # carrier lifetime (s)

# ---------- 2. PHYSICS FUNCTIONS ----------
def intrinsic_conc(T):
    """ni rises steeply with temperature."""
    return NI_300 * (T / 300) ** 1.5 * np.exp(-(EG / (2 * K_B)) * (1 / T - 1 / 300))

def built_in_potential(NA, ND, T):
    Vt = K_B * T
    ni = intrinsic_conc(T)
    return Vt * np.log(NA * ND / ni**2)

def depletion_width(NA, ND, Vbi, Vj):
    """Total width W in cm. Vj = voltage across the junction itself."""
    Vj = min(Vj, Vbi - 0.01)            # avoid sqrt of negative number
    return np.sqrt(2 * EPS_SI * (Vbi - Vj) / Q * (1 / NA + 1 / ND))

def saturation_current(NA, ND, T):
    ni = intrinsic_conc(T)
    Ln, Lp = np.sqrt(DN * TAU), np.sqrt(DP * TAU)
    return AREA * Q * ni**2 * (DP / (Lp * ND) + DN / (Ln * NA))

def safe_exp(x):
    return np.exp(np.clip(x, -700, 700))

def iv_curve(Is, n, T, Rs, Vbr):
    """Return junction voltage Vd, total voltage V and current I arrays."""
    Vt = K_B * T
    Vd = np.unique(np.concatenate([
        np.linspace(-Vbr - 0.6, -1, 800),
        np.linspace(-1, 1.0, 3200)]))
    I = Is * (safe_exp(Vd / (n * Vt)) - 1) - Is * safe_exp(-(Vd + Vbr) / (n * Vt))
    V = Vd + I * Rs                     # series resistance adds a voltage drop
    return Vd, V, I

def band_diagram(NA, ND, T, Vbi, Vj, n_pts=400):
    """Conduction band, valence band and Fermi levels (eV) vs x (nm)."""
    Vt = K_B * T
    ni = intrinsic_conc(T)
    W = depletion_width(NA, ND, Vbi, Vj)
    xn = W * NA / (NA + ND)
    xp = W * ND / (NA + ND)
    B = Vbi - min(Vj, Vbi - 0.01)       # barrier height
    drop_p = B * xp / W
    drop_n = B * xn / W

    x = np.linspace(-2 * W, 2 * W, n_pts)
    Ec = np.where(x < -xp, B,
         np.where(x < 0, B - drop_p * ((x + xp) / xp) ** 2,
         np.where(x < xn, drop_n * ((xn - x) / xn) ** 2, 0.0)))
    Ev = Ec - EG

    Ef_p = B - EG / 2 - Vt * np.log(NA / ni)    # p-side Fermi level
    Ef_n = -EG / 2 + Vt * np.log(ND / ni)       # n-side Fermi level
    return x * 1e7, Ec, Ev, Ef_p, Ef_n, xp * 1e7, xn * 1e7, W * 1e7

# ---------- 3. WEB PAGE LAYOUT ----------
st.set_page_config(page_title="PN Junction Simulator", layout="wide")
st.title("PN Junction Diode Simulator")
st.caption("Change the sliders and watch the physics respond in real time.")

# ---------- 4. SIDEBAR CONTROLS ----------
st.sidebar.header("Controls")
exp_A = st.sidebar.slider("Acceptor doping N_A (10^x cm^-3)", 14.0, 19.0, 16.0, 0.1)
exp_D = st.sidebar.slider("Donor doping N_D (10^x cm^-3)", 14.0, 19.0, 16.0, 0.1)
T = st.sidebar.slider("Temperature (K)", 250, 450, 300, 5)
n_id = st.sidebar.slider("Ideality factor n", 1.0, 2.0, 1.0, 0.05)
Rs = st.sidebar.slider("Series resistance Rs (ohm)", 0.0, 20.0, 0.0, 0.5)
Vbr = st.sidebar.slider("Breakdown voltage (V)", 5.0, 50.0, 20.0, 1.0)
V_applied = st.sidebar.slider("Applied voltage V (V)", -float(Vbr), 1.0, 0.0, 0.01)

NA, ND = 10 ** exp_A, 10 ** exp_D

# ---------- 5. CALCULATIONS ----------
Vbi = built_in_potential(NA, ND, T)
Is = saturation_current(NA, ND, T)
Vd, Vtot, I = iv_curve(Is, n_id, T, Rs, Vbr)

I_op = np.interp(V_applied, Vtot, I)          # current at slider voltage
Vj_op = np.interp(V_applied, Vtot, Vd)        # voltage across junction only
W_nm = depletion_width(NA, ND, Vbi, Vj_op) * 1e7

# ---------- 6. NUMBERS PANEL ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Built-in potential", f"{Vbi:.3f} V")
c2.metric("Depletion width", f"{W_nm:.1f} nm")
c3.metric("Saturation current", f"{Is:.2e} A")
c4.metric("Current at V", f"{I_op * 1e3:.4g} mA")

# ---------- 7. I-V PLOTS ----------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

mask = (Vtot > -1) & (Vtot < 1.2) & (I * 1e3 < 50)
ax1.plot(Vtot[mask], I[mask] * 1e3, lw=2)
ax1.plot(V_applied, I_op * 1e3, "ro", ms=9)
ax1.set(xlabel="Voltage (V)", ylabel="Current (mA)",
        title="I-V curve (linear)", ylim=(-2, 50))
ax1.axhline(0, color="gray", lw=0.5)
ax1.axvline(0, color="gray", lw=0.5)
ax1.grid(alpha=0.3)

ax2.semilogy(Vtot, np.abs(I), lw=2)
ax2.plot(V_applied, max(abs(I_op), 1e-20), "ro", ms=9)
ax2.set(xlabel="Voltage (V)", ylabel="|Current| (A)",
        title="I-V curve (log scale, shows breakdown)")
ax2.grid(alpha=0.3, which="both")
st.pyplot(fig1)

# ---------- 8. ENERGY BAND DIAGRAM ----------
x, Ec, Ev, Ef_p, Ef_n, xp, xn, W = band_diagram(NA, ND, T, Vbi, Vj_op)
fig2, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(x, Ec, "b", lw=2, label="Conduction band (Ec)")
ax3.plot(x, Ev, "g", lw=2, label="Valence band (Ev)")
ax3.hlines(Ef_p, x[0], 0, colors="r", linestyles="--", label="Fermi level")
ax3.hlines(Ef_n, 0, x[-1], colors="r", linestyles="--")
ax3.axvspan(-xp, xn, color="orange", alpha=0.2, label="Depletion region")
ax3.axvline(0, color="k", lw=0.8)
ax3.text(x[0] * 0.8, Ec.max() + 0.1, "P side")
ax3.text(x[-1] * 0.7, Ec.max() + 0.1, "N side")
ax3.set(xlabel="Position x (nm)", ylabel="Energy (eV)",
        title="Energy band diagram")
ax3.legend(loc="lower left")
ax3.grid(alpha=0.3)
st.pyplot(fig2)