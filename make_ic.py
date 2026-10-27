#!/usr/bin/env python3
"""Initial conditions for a two-galaxy collision.

Internal units set G = 1:
    [L] = 1 kpc
    [M] = 1e10 Msun
    [V] = 207.4 km/s
    [T] = 4.715 Myr

Each galaxy is a Plummer sphere (bulge + dark halo, sampled exactly from its
isotropic distribution function) plus an exponential stellar disk. The pair is
placed on a parabolic encounter orbit with a chosen pericentre.

Output is a flat binary file:
    int64   N
    float32 pos[N][3]
    float32 vel[N][3]
    float32 mass[N]
    int32   type[N]      0,1 = halo,disk of galaxy A;  2,3 = galaxy B
"""
import argparse
import numpy as np

HALO_M, HALO_A = 40.0, 20.0          # 4e11 Msun, 20 kpc
DISK_M, DISK_RD, DISK_ZD = 5.0, 3.5, 0.3


def _directions(k, rng):
    u = rng.uniform(-1.0, 1.0, k)
    phi = rng.uniform(0.0, 2.0 * np.pi, k)
    s = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    return np.stack([s * np.cos(phi), s * np.sin(phi), u], axis=1)


def sample_plummer(n, M, a, rng):
    """Exact isotropic Plummer sampling (Aarseth, Henon & Wielen 1974)."""
    m = rng.uniform(1e-10, 1.0, n)
    r = a / np.sqrt(m ** (-2.0 / 3.0) - 1.0)
    pos = r[:, None] * _directions(n, rng)

    # speeds by rejection on g(q) = q^2 (1 - q^2)^{7/2}
    q = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        x = rng.uniform(0.0, 1.0, todo.size)
        y = rng.uniform(0.0, 0.1, todo.size)
        ok = y < x * x * (1.0 - x * x) ** 3.5
        q[todo[ok]] = x[ok]
        todo = todo[~ok]

    v_esc = np.sqrt(2.0 * M / a) * (1.0 + r * r / (a * a)) ** (-0.25)
    vel = (q * v_esc)[:, None] * _directions(n, rng)
    return pos, vel


def sample_disk(n, M, Rd, zd, rng):
    """Exponential disk, circular orbits from the enclosed mass, warm enough
    to avoid immediate fragmentation."""
    # p(R) ~ R exp(-R/Rd) is exactly Gamma(shape=2, scale=Rd)
    R = rng.gamma(2.0, Rd, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    # sech^2 vertical profile
    u = rng.uniform(-0.999, 0.999, n)
    z = zd * np.arctanh(u)

    pos = np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)

    r = np.sqrt(R * R + z * z)
    m_halo = HALO_M * r ** 3 / (r * r + HALO_A * HALO_A) ** 1.5
    m_disk = M * (1.0 - (1.0 + R / Rd) * np.exp(-R / Rd))
    v_c = np.sqrt(np.maximum(1e-12, (m_halo + m_disk) / np.maximum(R, 1e-6)))

    sig_R, sig_z = 0.15 * v_c, 0.10 * v_c
    v_phi = v_c + sig_R * rng.standard_normal(n)
    v_R = sig_R * rng.standard_normal(n)
    vel = np.stack([
        v_R * np.cos(phi) - v_phi * np.sin(phi),
        v_R * np.sin(phi) + v_phi * np.cos(phi),
        sig_z * rng.standard_normal(n),
    ], axis=1)
    return pos, vel


def rotation(inc_deg, pa_deg):
    i, p = np.radians(inc_deg), np.radians(pa_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
    Rz = np.array([[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]])
    return Rz @ Rx


def build_galaxy(n_halo, n_disk, inc, pa, rng):
    ph, vh = sample_plummer(n_halo, HALO_M, HALO_A, rng)
    pd, vd = sample_disk(n_disk, DISK_M, DISK_RD, DISK_ZD, rng)
    Rm = rotation(inc, pa)
    pos = np.vstack([ph, pd @ Rm.T])
    vel = np.vstack([vh, vd @ Rm.T])
    mass = np.concatenate([np.full(n_halo, HALO_M / n_halo),
                           np.full(n_disk, DISK_M / n_disk)])
    kind = np.concatenate([np.zeros(n_halo, np.int32), np.ones(n_disk, np.int32)])
    return pos, vel, mass, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600000, help="total particle count")
    ap.add_argument("--disk-frac", type=float, default=0.5)
    ap.add_argument("--sep", type=float, default=60.0, help="initial separation, kpc")
    ap.add_argument("--rperi", type=float, default=12.0, help="pericentre, kpc")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="ic.bin")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    per_gal = args.n // 2
    n_disk = int(per_gal * args.disk_frac)
    n_halo = per_gal - n_disk

    # Galaxy A nearly prograde to the orbit, galaxy B inclined.
    pA, vA, mA, kA = build_galaxy(n_halo, n_disk, inc=20.0, pa=0.0, rng=rng)
    pB, vB, mB, kB = build_galaxy(n_halo, n_disk, inc=70.0, pa=45.0, rng=rng)

    # Parabolic two-body orbit: v = sqrt(2 G Mtot / d), pericentre from L.
    Mgal = HALO_M + DISK_M
    Mtot = 2.0 * Mgal
    d = args.sep
    v_par = np.sqrt(2.0 * Mtot / d)
    v_t = np.sqrt(2.0 * Mtot * args.rperi) / d
    v_r = -np.sqrt(max(0.0, v_par * v_par - v_t * v_t))

    pA += np.array([-d / 2, 0.0, 0.0])
    pB += np.array([+d / 2, 0.0, 0.0])
    vA += np.array([-v_r / 2, -v_t / 2, 0.0])
    vB += np.array([+v_r / 2, +v_t / 2, 0.0])

    pos = np.vstack([pA, pB]).astype(np.float32)
    vel = np.vstack([vA, vB]).astype(np.float32)
    mass = np.concatenate([mA, mB]).astype(np.float32)
    kind = np.concatenate([kA, kB + 2]).astype(np.int32)

    # Remove any net drift so the pair stays centred in the frame.
    vcom = (vel * mass[:, None]).sum(0) / mass.sum()
    vel -= vcom.astype(np.float32)

    n = pos.shape[0]
    with open(args.out, "wb") as f:
        np.array([n], dtype=np.int64).tofile(f)
        pos.tofile(f); vel.tofile(f); mass.tofile(f); kind.tofile(f)

    print(f"wrote {args.out}: N={n}  ({n_halo} halo + {n_disk} disk per galaxy)")
    print(f"  Mgal={Mgal:.1f}e10 Msun  sep={d} kpc  rperi={args.rperi} kpc")
    print(f"  v_par={v_par:.4f}  v_r={v_r:.4f}  v_t={v_t:.4f}  [207.4 km/s]")


if __name__ == "__main__":
    main()
