# Recommended CTF estimate (untilted 300 kV movies)

These settings matched **CTFFind5 defocus and astigmatism** best on a
5-micrograph no-laser smoke set (native pixel **1.059 Å**; CTFFind5 box 512,
fit 30–5 Å). Use them as the default for untilted 300 kV movies.

**Thickness is a separate estimator.** PICASSO reports a 1D CTF\(_t\) fit
(grid + thickness-only Adam). CTFFind5 seeds ice from a GoF first-node
(\(t \approx d_{\mathrm{GoF}}^2 / \lambda\)). The two should not be expected
to agree. On the smoke set, PICASSO was systematically ~170–270 Å thicker
than CTFFind; treat `thickness_angstroms` as our CTF\(_t\) number, not as
validated ice thickness.

Copy `ctf_estimate_example_config.yaml` and point it at your micrographs.

## Pipeline (do this, in this order)

1. **1D defocus** on the mean power spectrum.
2. **2D defocus** (astigmatism + spatial field on patches). **This is the
   reported defocus** (`defocus_u` / `defocus_v` / mean in the optics CSV).
3. **1D thickness grid** (300–4000 Å, 100 Å step) on a **tilt-corrected +
   equiphase** mean spectrum, using the **2D mean defocus**, log-spline
   background.
4. **Adam on thickness only** (`refine_dim: "thickness"`). Defocus stays
   frozen at the 2D result.

Do **not** jointly refine defocus with thickness (`refine_dim: "1d"` or
`"2d"`). On the no-laser set that made ice systematically too thin and
pulled mean defocus away from CTFFind5.

## YAML knobs that matter

| Knob | Value | Why |
|---|---|---|
| `target_pixel_spacing_angstroms` | `1.4` | Nyquist 2.8 Å, so a 4 Å thickness cutoff still includes the first CTF\(_t\) node (~4.8–5.6 Å). At 3.0 Å, Nyquist is 6 Å and the 4 Å cutoff is clipped — thickness then fits envelope only and comes out ~300–400 Å too thick. |
| `fitting_config.frequency_fit_range_angstroms` | `[30, 4]` | Defocus band. Same as thickness. |
| `thickness_config.frequency_fit_range_angstroms` | `[30, 4]` | Thickness needs the first CTF\(_t\) node (~4.8–5.6 Å). |
| `patch_sidelength` | `512` | Match a typical CTFFind box. |
| `defocus_grid_resolution` | `[1, 3, 3]` | Gentle spatial field (untilted movies). |
| `optimize_envelope_1d` | `false` | Envelope ON competed with thickness. |
| `use_tilt_corrected_ps` | `true` | Warp patches to mean defocus before 1D thickness. |
| `use_equiphase` | `true` | Astigmatism does not smear Thon rings in the 1D profile. |
| `refine_dim` | `"thickness"` | Grid + t-only Adam. Freeze 2D defocus. |
| Background | log-spline (default) | Cubic poly+0.5 (CTFFind-style) was systematically too thin. |

## What not to use (tried, dropped)

- **`refine_dim: "2d"`** — ice collapsed ~400–700 Å thin vs CTFFind; joint defocus drifted off.
- **`refine_dim: "1d"`** — helped one micrograph’s thickness a little, hurt defocus (hundreds of Å).
- **Fitting pixel 3.0 Å with a 4 Å cutoff** — thickness node not in band.
- **1D envelope (`optimize_envelope_1d: true`)** — trades against ice thickness.
- **GoF / first-node seeding** — seeds were far too thick (fitG 6–8 Å vs CTFFind 4.3–5.7 Å). Full 300–4000 Å grid is better.
- **Poly detrend for thickness scoring** — bracketed CTFFind from the thin side; spline from the thick side. Spline stayed.
- **2D amplitude vs \|CTF\|** — mean defocus slightly worse; modest astigmatism poorly constrained.

## Smoke-set vs CTFFind5 (ignore 0029)

0029 has no 2DTM CSV and is an outlier (PICASSO thickness ~half of CTFFind;
CTFFind CC 0.07). Numbers below are from the recommended route (both bands
`[30,4]`, t-only Adam, 1.4 Å).

| mgraph | df_2d | df_CTFFind | thick_P | thick_C |
|--------|------:|-----------:|--------:|--------:|
| 0007 | 12010 | 12051 | 1329 | 1157 |
| 0071 | 11923 | 11925 | 1524 | 1334 |
| 0164 | 11854 | 11868 | 1868 | 1601 |
| 0190 | 12238 | 12059 | 1576 | 1375 |

Mean defocus tracks CTFFind closely (0071/0164 almost exact; 0007 −41 Å).
Thickness is consistently thicker than CTFFind (different estimator).

## Outputs

Per dataset:

- `ctf_estimate_optics.csv` — Leopard-EM columns; **use this defocus**
- `ctf_estimate_metrics.csv` — thickness, staged defocus, CCs, spatial field

Per micrograph (if enabled):

- `*_optics.yaml` — Leopard optics group + thickness + defocus field
- `*_debug.npz` — 1D spectra, thickness CC curve, mean PS
- `*_ctf_plots_1d.png`, `*_ctf_plots_2d.png`, `*_ctf_plots_thickness_cc.png`

Optics CSV `defocus_u`/`defocus_v` are the 2D result. Thickness is
`thickness_angstroms` in the metrics CSV / per-image YAML.

## Command

From the PICASSO repo root. Always `uv run --no-sync`.

```bash
# Edit YAML_CONFIG_PATH in the program, or copy the example YAML.
uv run --no-sync python programs/ctf_estimation/run_ctf_estimation.py
```

Example YAML (same knobs as this document):
`programs/ctf_estimation/ctf_estimate_example_config.yaml`.
