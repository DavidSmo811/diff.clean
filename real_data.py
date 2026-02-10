import subprocess

def get_least_used_gpu():
    output = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
        shell=True
    )
    mem_usage = [int(x) for x in output.decode().strip().split("\n")]
    return min(range(len(mem_usage)), key=lambda i: mem_usage[i])

gpu_id = get_least_used_gpu()
print(f"Selected GPU: {gpu_id}")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import torch
from casacore.tables import table
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from unet_film2 import UNet
from inference import InferenceLoop
from pathlib import Path
from matplotlib.colors import PowerNorm
from astropy.wcs import WCS
import argparse


# def read_ms(msname, field_id=0, device="cpu"):
#     tab = table(msname, readonly=True)

#     field_id_np = tab.getcol("FIELD_ID")
#     sel = (field_id_np == field_id)
#     print("Selected FIELD_ID rows:", np.sum(sel))

#     if not np.any(sel):
#         raise ValueError(f"No rows found for FIELD_ID={field_id}")

#     # ---------- CPU: MS I/O ----------
#     try:
#         vis_np  = tab.getcol("CORRECTED_DATA")[sel]   # (nrow, nchan, ncorr)
#     except Exception:
#         vis_np  = tab.getcol("DATA")[sel]             # (nrow, nchan, ncorr)
#         print("No CORRECTED_DATA found; using DATA instead.")
#     flag_np = tab.getcol("FLAG")[sel]
#     uvw_m   = tab.getcol("UVW")[sel]

#     try:
#         weights_np = tab.getcol("WEIGHT_SPECTRUM")[sel]
#         weights_np = np.reshape(weights_np, vis_np.shape)
#         use_weights = True
#         print("Using WEIGHT_SPECTRUM for averaging.")
#     except Exception:
#         use_weights = False
#         print("No WEIGHT_SPECTRUM found; using uniform weights.")

#     tab.close()

#     # ---------- GPU ----------
#     vis = torch.as_tensor(vis_np, dtype=torch.complex64, device=device)
#     flag = torch.as_tensor(flag_np, dtype=torch.bool, device=device)

#     # Flags: OR über XX / YY
#     flag_stokesI = flag[:, :, 0] | flag[:, :, 3]
#     vis = torch.where(flag_stokesI.unsqueeze(-1), torch.zeros_like(vis), vis)

#     if use_weights:
#         weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
#         weights = weights + 1e-10

#         # gewichtetes Channel-Averaging pro Korrelation
#         vis = torch.sum(vis * weights, dim=1) / torch.sum(weights, dim=1)
#         # -> (nrow, ncorr)
#     else:
#         vis = torch.mean(vis, dim=1)

#     # Stokes I NACH dem Channel-Average
#     vis = 0.5 * (vis[:, 0] + vis[:, 3])
#     vis = vis.flatten() #* vis_scale

#     # ---------- UVW ----------
#     spw = table(msname + "/SPECTRAL_WINDOW")
#     freq = spw.getcol("CHAN_FREQ")[0]
#     spw.close()

#     field = table(msname + "/FIELD")
#     phase_dir = field.getcol("PHASE_DIR")[0,0]
#     ra0, dec0 = phase_dir
#     field.close()

#     nu = float(np.mean(freq))
#     c = 299792458.0
#     wavelengths = c / nu

#     uvw = torch.as_tensor(uvw_m, dtype=torch.float32, device=device)
#     uvw = uvw.T.unsqueeze(0) / wavelengths  # (1, 3, nrow)

#     return vis, uvw, nu, ra0, dec0

# def read_ms(msname, field_id=0, channel_id=20, device="cpu"):
#     """
#     Liest MS und verwendet nur EINEN Channel (kein Averaging).
    
#     Parameters:
#     -----------
#     channel_id : int
#         Welcher Channel soll verwendet werden (default: 0 = erster Channel)
#     """
#     tab = table(msname, readonly=True)
    
#     field_id_np = tab.getcol("FIELD_ID")
#     sel = (field_id_np == field_id)
    
#     if not np.any(sel):
#         raise ValueError(f"No rows found for FIELD_ID={field_id}")
    
#     # ---------- CPU: MS I/O ----------
#     try:
#         vis_np = tab.getcol("CORRECTED_DATA")[sel]   # (nrow, nchan, ncorr)
#     except Exception:
#         vis_np = tab.getcol("DATA")[sel]
#         print("No CORRECTED_DATA found; using DATA instead.")
    
#     flag_np = tab.getcol("FLAG")[sel]
#     uvw_m = tab.getcol("UVW")[sel]
    
#     tab.close()
    
#     # ---------- KEIN CHANNEL-AVERAGING! ----------
#     # Wähle nur EINEN Channel aus
#     nchan = vis_np.shape[1]
#     print(f"Total channels available: {nchan}")
#     print(f"Using ONLY channel {channel_id}")
    
#     vis_np = vis_np[:, channel_id, :]      # (nrow, ncorr)
#     flag_np = flag_np[:, channel_id, :]    # (nrow, ncorr)
    
#     # ---------- GPU ----------
#     vis = torch.as_tensor(vis_np, dtype=torch.complex64, device=device)
#     flag = torch.as_tensor(flag_np, dtype=torch.bool, device=device)
    
#     # Flags: OR über XX / YY
#     flag_stokesI = flag[:, 0] | flag[:, 3]
    
#     # Setze geflaggte Visibilities auf 0
#     vis[flag_stokesI, :] = 0.0
    
#     # Stokes I (ohne Averaging über Channels!)
#     vis = 0.5 * (vis[:, 0] + vis[:, 3])
#     vis = vis.flatten()
    
#     # ---------- UVW ----------
#     spw = table(msname + "/SPECTRAL_WINDOW")
#     freq = spw.getcol("CHAN_FREQ")[0]  # Alle Frequenzen
#     spw.close()
    
#     field = table(msname + "/FIELD")
#     phase_dir = field.getcol("PHASE_DIR")[0, 0]
#     ra0, dec0 = phase_dir
#     field.close()
    
#     # Verwende die EXAKTE Frequenz des gewählten Channels!
#     nu = float(freq[channel_id])
#     print(f"Channel {channel_id} frequency: {nu/1e9:.3f} GHz")
    
#     c = 299792458.0
#     wavelengths = c / nu
    
#     uvw = torch.as_tensor(uvw_m, dtype=torch.float32, device=device)
#     uvw = uvw.T.unsqueeze(0) / wavelengths  # (1, 3, nrow)
    
#     return vis, uvw, nu, ra0, dec0

# def read_ms(msname, field_id=0, device="cpu"):
#     tab = table(msname, readonly=True)

#     sel = tab.getcol("FIELD_ID") == field_id
#     if not np.any(sel):
#         raise ValueError("No rows for FIELD_ID")

#     # --- DATA ---
#     try:
#         vis_np = tab.getcol("CORRECTED_DATA")[sel]   # (nrow, nchan, ncorr)
#     except Exception:
#         vis_np = tab.getcol("DATA")[sel]

#     flag_np = tab.getcol("FLAG")[sel]
#     uvw_m = tab.getcol("UVW")[sel]                  # (nrow, 3)
#     tab.close()

#     nrow, nchan, ncorr = vis_np.shape

#     # --- Frequencies ---
#     spw = table(msname + "/SPECTRAL_WINDOW")
#     freq = spw.getcol("CHAN_FREQ")[0]               # (nchan,)
#     spw.close()

#     # --- Phase center ---
#     field = table(msname + "/FIELD")
#     ra0, dec0 = field.getcol("PHASE_DIR")[0, 0]
#     field.close()

#     # --- Torch ---
#     vis = torch.as_tensor(vis_np, dtype=torch.complex64, device=device)
#     flag = torch.as_tensor(flag_np, dtype=torch.bool, device=device)

#     # --- Stokes I flags ---
#     flag_I = flag[..., 0] | flag[..., 3]
#     vis[flag_I] = 0.0

#     # --- Stokes I ---
#     vis = 0.5 * (vis[..., 0] + vis[..., 3])   # (nrow, nchan)

#     # --- Flatten visibilities ---
#     vis = vis.reshape(-1)                     # (nrow * nchan)

#     # --- UVW per channel ---
#     c = 299792458.0
#     wavelengths = c / torch.tensor(freq, device=device)

#     uvw = torch.as_tensor(uvw_m, device=device)     # (nrow, 3)
#     uvw = uvw[:, None, :] / wavelengths[None, :, None]
#     uvw = uvw.reshape(-1, 3).T.unsqueeze(0)         # (1, 3, nvis)

#     nu_ref = float(freq.mean())  # nur noch Meta-Info

#     return vis, uvw, nu_ref, ra0, dec0

def read_ms(msname, field_id=0, num_channel_batches=32, device="cpu"):
    """
    Teile Channels in Batches auf und average innerhalb jedes Batches.
    
    Parameters:
    -----------
    num_channel_batches : int
        Anzahl der Channel-Gruppen. 
        Z.B. bei 512 Channels und 8 Batches → 64 Channels pro Batch
        
    Returns:
    --------
    vis_list : list of torch.Tensor
        Liste mit Visibilities für jeden Channel-Batch
    uvw_list : list of torch.Tensor  
        Liste mit UVW-Koordinaten für jeden Channel-Batch
    nu_list : list of float
        Frequenzen (Mittelwert pro Batch)
    """
    tab = table(msname, readonly=True)
    
    field_id_np = tab.getcol("FIELD_ID")
    sel = (field_id_np == field_id)
    
    if not np.any(sel):
        raise ValueError(f"No rows found for FIELD_ID={field_id}")
    
    # Lade Daten
    try:
        vis_np = tab.getcol("CORRECTED_DATA")[sel]
    except Exception:
        vis_np = tab.getcol("DATA")[sel]
        print("No CORRECTED_DATA found; using DATA instead.")
    
    flag_np = tab.getcol("FLAG")[sel]
    uvw_m = tab.getcol("UVW")[sel]
    
    try:
        weights_np = tab.getcol("WEIGHT_SPECTRUM")[sel]
        weights_np = np.reshape(weights_np, vis_np.shape)
        use_weights = True
        print("Using WEIGHT_SPECTRUM for averaging.")
    except Exception:
        weights_np = None
        use_weights = False
        print("No WEIGHT_SPECTRUM found; using uniform weights.")
    
    tab.close()
    
    # Metadaten
    spw = table(msname + "/SPECTRAL_WINDOW")
    freq = spw.getcol("CHAN_FREQ")[0]
    spw.close()
    
    field = table(msname + "/FIELD")
    phase_dir = field.getcol("PHASE_DIR")[0, 0]
    ra0, dec0 = phase_dir
    field.close()
    
    c = 299792458.0
    nrow, nchan, ncorr = vis_np.shape
    
    # Teile Channels in Batches
    channels_per_batch = nchan // num_channel_batches
    if nchan % num_channel_batches != 0:
        channels_per_batch += 1
    
    print(f"\nChannel Batching:")
    print(f"  Total channels: {nchan}")
    print(f"  Number of batches: {num_channel_batches}")
    print(f"  Channels per batch: ~{channels_per_batch}")
    
    vis_list = []
    uvw_list = []
    nu_list = []
    
    for batch_idx in range(num_channel_batches):
        # Channel-Range für diesen Batch
        ch_start = batch_idx * channels_per_batch
        ch_end = min((batch_idx + 1) * channels_per_batch, nchan)
        
        if ch_start >= nchan:
            break
        
        print(f"\n  Processing batch {batch_idx+1}/{num_channel_batches}: channels {ch_start}-{ch_end-1}")
        
        # Slice für diesen Batch
        vis_batch = vis_np[:, ch_start:ch_end, :]      # (nrow, n_ch_batch, ncorr)
        flag_batch = flag_np[:, ch_start:ch_end, :]
        freq_batch = freq[ch_start:ch_end]
        
        if use_weights:
            weights_batch = weights_np[:, ch_start:ch_end, :]
        
        # Mittlere Frequenz dieses Batches
        nu_batch = float(np.mean(freq_batch))
        lambda_batch = c / nu_batch
        
        # UVW für diese mittlere Frequenz
        uvw_batch = uvw_m / lambda_batch
        
        # Zu Torch
        vis_t = torch.as_tensor(vis_batch, dtype=torch.complex64, device=device)
        flag_t = torch.as_tensor(flag_batch, dtype=torch.bool, device=device)
        
        # Flags: OR über XX / YY
        flag_stokesI = flag_t[:, :, 0] | flag_t[:, :, 3]  # (nrow, n_ch_batch)
        
        # Setze geflaggte auf 0
        vis_t = torch.where(flag_stokesI.unsqueeze(-1), torch.zeros_like(vis_t), vis_t)
        
        # Gewichtetes Averaging über Channels in diesem Batch
        if use_weights:
            weights_t = torch.as_tensor(weights_batch, dtype=torch.float32, device=device)
            weights_t = weights_t + 1e-10
            
            # Gewichtetes Averaging pro Korrelation
            vis_t = torch.sum(vis_t * weights_t, dim=1) / torch.sum(weights_t, dim=1)
        else:
            vis_t = torch.mean(vis_t, dim=1)
        
        # Stokes I
        vis_stokes = 0.5 * (vis_t[:, 0] + vis_t[:, 3])  # (nrow,)
        
        # Zu Liste hinzufügen
        vis_list.append(vis_stokes.unsqueeze(0))  # (1, nrow)
        
        uvw_t = torch.as_tensor(uvw_batch, dtype=torch.float32, device=device)
        uvw_list.append(uvw_t.T.unsqueeze(0))  # (1, 3, nrow)
        
        nu_list.append(nu_batch)
        
        print(f"    Frequency: {nu_batch/1e9:.3f} GHz")
        print(f"    Visibilities: {vis_stokes.shape[0]}")
    
    return vis_list, uvw_list, nu_list, ra0, dec0


# def make_lmn_grid(npix, res_in_arcsec, dec=0, device="cpu"):
#     """
#     Erstellt ein (l,m,n) Grid basierend auf pyvisgen approach
    
#     Parameters:
#     -----------
#     npix : int
#         Anzahl der Pixel pro Achse
#     res_in_arcsec : float
#         Pixelgröße in arcsec
#     dec : float
#         Deklination in Radiant (für Projektionskorrektur)
    
#     Returns:
#     --------
#     lmn : np.ndarray
#         Array mit Shape (3, npix, npix) mit [L, M, N-1]
#     """
#     # FOV in Radiant
#     fov_rad = (npix * res_in_arcsec / 3600.0) * np.pi / 180.0#Eigentlich npix statt npix/2
    
#     # Erstelle lineare Achsen von -fov/2 bis +fov/2
#     axis = np.linspace(-fov_rad / 2, fov_rad / 2, npix)
    
#     # Meshgrid für L und M
#     L, M = np.meshgrid(axis, axis, indexing='xy')

#     # N berechnen: n = sqrt(1 - l^2 - m^2)  
#     # Für den w-term wird oft (n-1) verwendet
#     lm2 = L**2 + M**2
#     N = np.sqrt(np.maximum(0.0, 1.0 - lm2))#np.ones_like(L)#np.sqrt(np.maximum(0.0, 1.0 - lm2)) - 1.0  # -1 für w-term
    
#     # Stack zu (3, npix, npix)
#     lmn = np.stack([L, M, N], axis=0)
#     print("L,M,N shape:", lmn.shape)
    
#     lmn_torch = torch.tensor(lmn, dtype=torch.float32, device=device)
    
#     # Reshape: (3, npix, npix) -> (3, npix*npix)
#     lmn_flat = lmn_torch.reshape(3, -1)
    
#     # Batch dimension: (3, npix*npix) -> (1, 3, npix*npix)
#     lmn_batched = lmn_flat.unsqueeze(0)
    
#     print(f"\nLMN prepared for inference:")
#     print(f"  Shape: {lmn_batched.shape}")
#     print(f"  L: min={lmn_batched[0, 0].min():.6f}, max={lmn_batched[0, 0].max():.6f}, unique={len(torch.unique(lmn_batched[0, 0]))}")
#     print(f"  M: min={lmn_batched[0, 1].min():.6f}, max={lmn_batched[0, 1].max():.6f}, unique={len(torch.unique(lmn_batched[0, 1]))}")
#     print(f"  N: min={lmn_batched[0, 2].min():.6f}, max={lmn_batched[0, 2].max():.6f}, unique={len(torch.unique(lmn_batched[0, 2]))}")
    
#     return lmn_batched

def make_lmn_grid(npix, res_in_arcsec, dec=0, device="cpu"):
    """
    Erstellt ein (l,m,n) Grid nach pyvisgen-Konvention.
    
    Parameters:
    -----------
    npix : int
        Anzahl der Pixel pro Achse
    res_in_arcsec : float
        Pixelgröße in arcsec
    dec : float
        Deklination des Phasenzentrums in GRAD
    
    Returns:
    --------
    lmn_batched : torch.Tensor
        Shape (1, 3, npix*npix) mit [L, M, N]
    """
    # FOV in Radiant
    fov_rad = (npix * res_in_arcsec / 3600.0) * np.pi / 180.0
    res_rad = fov_rad / npix
    
    # Deklination in Radiant (wichtig für sphärische Projektion!)
    dec_rad = np.deg2rad(dec).astype(np.float64)
    
    # RA/DEC Grid (wie pyvisgen)
    # Achtung: np.arange stoppt VOR dem stop-Wert!
    r = np.arange(
        start=-(npix / 2) * res_rad,
        stop=(npix / 2) * res_rad,
        step=res_rad,
        dtype=np.float64
    )
    
    # Debug: Prüfe ob wir exakt npix Punkte haben
    assert len(r) == npix, f"Grid hat {len(r)} Punkte statt {npix}!"
    
    d = r + dec_rad  # Absolute Deklination
    
    # Meshgrid
    R, D = np.meshgrid(r, d, indexing='xy')
    
    # Sphärische Projektion (EXAKT wie pyvisgen!)
    L = np.cos(D) * np.sin(R)
    M = np.sin(D) * np.cos(dec_rad) - np.cos(D) * np.sin(dec_rad) * np.cos(R)
    
    # N berechnen (OHNE -1, das macht Finufft!)
    lm2 = L**2 + M**2
    N = np.sqrt(np.maximum(0.0, 1.0 - lm2))
    
    # Debugging
    print(f"\nLMN Grid (pyvisgen style):")
    print(f"  FOV: {fov_rad * 180/np.pi * 3600:.1f} arcsec")
    print(f"  Pixel size: {res_rad * 180/np.pi * 3600:.3f} arcsec")
    print(f"  Grid points: {len(r)}")
    print(f"  r range: [{r.min():.6e}, {r.max():.6e}] rad")
    print(f"  L range: [{L.min():.6f}, {L.max():.6f}]")
    print(f"  M range: [{M.min():.6f}, {M.max():.6f}]")
    print(f"  N range: [{N.min():.6f}, {N.max():.6f}]")
    
    # Stack und reshape
    lmn = np.stack([L, M, N], axis=0)  # (3, npix, npix)
    lmn_torch = torch.tensor(lmn, dtype=torch.float32, device=device)
    lmn_batched = lmn_torch.reshape(3, -1).unsqueeze(0)  # (1, 3, npix*npix)
    
    return lmn_batched

def make_wcs(npix, res_arcsec, ra0_rad, dec0_rad):
    w = WCS(naxis=2)

    # Referenzpixel: Bildmitte (FITS ist 1-basiert!)
    crpix = (npix + 1) / 2

    # Pixelgröße in Grad
    cdelt = res_arcsec / 3600.0

    w.wcs.crpix = [crpix, crpix]
    w.wcs.cdelt = np.array([-cdelt, cdelt])   # RA negativ!
    w.wcs.crval = [
        np.degrees(ra0_rad),
        np.degrees(dec0_rad)
    ]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]

    return w

# def run_inference_from_ms(
#     msname,
#     infer,
#     npix=512,
#     res_in_arcsec=1.0,
#     device="cpu",
#     num_major_cycles=7,
#     vis_scale=1,
#     npz_name=None,
# ):
#     if npz_name is not None:
#         try:
#             print("Loading preprocessed data from NPZ...")
#             saved = np.load(npz_name)
#             vis  = torch.tensor(saved["vis"], device=device)
#             uvw  = torch.tensor(saved["uvw"], device=device)
#             nu   = saved["nu"].item()
#             ra0  = saved["ra0"].item()
#             dec0 = saved["dec0"].item()
#             print("Loaded preprocessed data.")
#         except Exception:
#             print("NPZ not found, reading MS...")
#             vis, uvw, nu, ra0, dec0 = read_ms(msname, device=device)
#             npz_path = Path(npz_name)
#             npz_path.parent.mkdir(parents=True, exist_ok=True)
#             np.savez(npz_name,
#                      vis=vis.cpu().numpy(),
#                      uvw=uvw.cpu().numpy(),
#                      nu=nu, ra0=ra0, dec0=dec0,)
#     else:
#         print("No NPZ specified, reading MS...")
#         vis, uvw, nu, ra0, dec0 = read_ms(msname, device=device)
#     wcs = make_wcs(npix, res_in_arcsec, ra0, dec0)
#     #uvw = uvw_to_lambda(uvw, nu)

#     print("Building LMN grid...")
#     lmn = make_lmn_grid(npix, res_in_arcsec,device=device)
    
#     # Torch
#     vis = torch.tensor(vis, dtype=torch.complex64, device=device)[None, :]*vis_scale
#     uvw = torch.tensor(uvw, dtype=torch.float32, device=device)#[None, :, :].T.unsqueeze(0)
#     #lmn = torch.flatten(torch.tensor(lmn, dtype=torch.float32, device=device)[None, ...],start_dim=2, end_dim=3)
#     print(lmn.shape)
#     print(vis.shape)
#     print(lmn[:,:,:10])

#     print("Running inference...")
#     result = infer.run_major_minor(
#         vis=vis,
#         uvw=uvw,
#         lmn=lmn,
#         num_major_cycles=num_major_cycles,
#     )

#     return result, wcs

def run_inference_from_ms(
    msname,
    infer,
    npix=512,
    res_in_arcsec=1.0,
    device="cpu",
    num_major_cycles=7,
    vis_scale=1,
    num_channel_batches=32,
    npz_name=None,
):
    """
    Inference mit Channel-Batching.
    """
    # Lade Daten in Batches
    if npz_name is not None and Path(npz_name).exists():
        print("Loading preprocessed batched data from NPZ...")
        saved = np.load(npz_name, allow_pickle=True)
        
        vis_list = [torch.tensor(v, device=device) for v in saved["vis_list"].tolist()]
        uvw_list = [torch.tensor(u, device=device) for u in saved["uvw_list"].tolist()]
        nu_list = saved["nu_list"].tolist()
        ra0 = saved["ra0"].item()
        dec0 = saved["dec0"].item()
        print(f"Loaded {len(vis_list)} channel batches from NPZ.")
    else:
        print("Reading MS with channel batching...")
        vis_list, uvw_list, nu_list, ra0, dec0 = read_ms(
            msname=msname,
            field_id=0,
            num_channel_batches=num_channel_batches,
            device=device
        )
        
        # NPZ speichern
        if npz_name is not None:
            print(f"Saving preprocessed batched data to {npz_name}...")
            npz_path = Path(npz_name)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(
                npz_name,
                vis_list=np.array([v.cpu().numpy() for v in vis_list], dtype=object),
                uvw_list=np.array([u.cpu().numpy() for u in uvw_list], dtype=object),
                nu_list=np.array(nu_list),
                ra0=ra0,
                dec0=dec0,
            )
            print(f"Saved {len(vis_list)} batches to NPZ.")
    
    # WCS (mit mittlerer Frequenz aller Batches)
    nu_mean = np.mean(nu_list)
    wcs = make_wcs(npix, res_in_arcsec, ra0, dec0)
    
    # LMN Grid (einmal für alle Batches)
    print("Building LMN grid...")
    lmn = make_lmn_grid(npix, res_in_arcsec, dec=np.degrees(dec0), device=device)
    
    print(f"\nRunning inference on {len(vis_list)} channel batches...")
    
    results_per_batch = []
    
    for batch_idx, (vis_batch, uvw_batch) in enumerate(zip(vis_list, uvw_list)):
        print(f"\n{'='*60}")
        print(f"Processing Channel Batch {batch_idx+1}/{len(vis_list)}")
        print(f"Frequency: {nu_list[batch_idx]/1e9:.3f} GHz")
        print(f"Visibilities: {vis_batch.shape[1]}")
        print(f"{'='*60}")
        
        # Skalierung
        vis_batch = vis_batch * vis_scale
        
        # Inference
        result = infer.run_major_minor(
            vis=vis_batch,
            uvw=uvw_batch,
            lmn=lmn,
            num_major_cycles=num_major_cycles,
        )
        
        results_per_batch.append(result)
    
    # KOMBINIERE die Ergebnisse
    print(f"\n{'='*60}")
    print(f"Combining results from {len(results_per_batch)} batches...")
    print(f"{'='*60}")
    
    # Weighted Average basierend auf Anzahl Visibilities
    model_images = []
    weights = []
    
    for batch_idx, result in enumerate(results_per_batch):
        model = result["model_image"][0, 0]
        model_images.append(model)
        
        # Gewicht: Anzahl der Visibilities (proportional zu SNR)
        weight = vis_list[batch_idx].shape[1]
        weights.append(weight)
    
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    weights = weights / weights.sum()  # Normalisiere
    
    print(f"Batch weights: {[f'{w:.3f}' for w in weights.cpu().numpy()]}")
    
    # Gewichteter Durchschnitt der Model Images
    combined_model = sum(w * img for w, img in zip(weights, model_images))
    
    # Kombiniere auch Residuals (für alle Major Cycles)
    num_cycles = len(results_per_batch[0]["res_image"])
    residual_images_combined = []
    
    for cycle_idx in range(num_cycles):
        residuals = [
            torch.as_tensor(r["res_image"][cycle_idx], device=device)
            for r in results_per_batch
        ]

        combined_res = torch.zeros_like(residuals[0])
        for w, res in zip(weights, residuals):
            combined_res += w * res

        residual_images_combined.append(combined_res)
    
    # Final result (gleiche Struktur wie vorher!)
    combined_result = {
        "model_image": combined_model.unsqueeze(0).unsqueeze(0),  # (1, 1, npix, npix)
        "res_image": residual_images_combined,  # Liste von Tensors
    }
    
    return combined_result, wcs

def read_json_config(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run inference on radio interferometric data using a trained UNet model.")
    argparser.add_argument('--config', type=str, required=False, default=None, help='Path to JSON config file.')
    args = argparser.parse_args()
    config=read_json_config(args.config)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIX_SIZE = config["pixsize"]#1200
    RES_IN_ARCSEC = config["res_in_arcsec"]#2#2.9#0.35
    FOV_ARCSEC = PIX_SIZE * RES_IN_ARCSEC
    NUM_MAJOR_CYCLES = config["num_major_cycles"]#7
    OBJ_NAME=config["obj_name"]#"n2820"
    MS_PATH=config["ms_path"]#f"/hs/fs08/data/group-brueggen/bav1814/sband/{OBJ_NAME}/{OBJ_NAME}_cs_o1_stokes_i.ms"
    SAVE_PATH=config["save_path"]#"./diff.clean_save/"
    VIS_SCALE=config["vis_scale"]#800
    CHECKPOINT_PATH = config["checkpoint_path"]#"/hs/babbage/data/group-brueggen/David/diff.clean_save/unet_reconstruction_new_combined_loss_new_scaler_maj7/unet_final_cycle_7.pt" #unet_reconstruction_LogHuber_Sky_image_is_loss_two_opt_loss_with_masking_res_no_abs/unet_final_cycle_5.pt"#"/hs/babbage/data/group-brueggen/David/diff.clean_save/unet_final_cycle_5.pt"
    NPZ_NAME = config["npz_name"]#"/hs/babbage/data/group-brueggen/David/preprocessed_Angelina_dataset.npz"
    FIELD_ID = config.get("field_id", 0)
    if NPZ_NAME is not None:
        NPZ_PATH=f"{SAVE_PATH}/{OBJ_NAME}/{NPZ_NAME}.npz"
    unet = UNet(
        num_conditions=NUM_MAJOR_CYCLES + 1,
        in_channels=2,
        out_channels=1,
        base_c=32,
        #LOFAR_scaling=False,
        #Conditional_Scaling=False,
    )

    infer = InferenceLoop(
        unet=unet,
        pix_size=PIX_SIZE,
        fov_arcsec=FOV_ARCSEC,
        device=DEVICE,
        use_wandb=False
    )

    infer.load_checkpoint(CHECKPOINT_PATH)
    
    result, wcs=run_inference_from_ms(
        msname=MS_PATH,#"/hs/babbage/data/shared/Angelina_Meerkat/obs25_1686240076_sdp_l0-virgo083-avg.ms.copy",#msname=f"/hs/fs08/data/group-brueggen/bav1814/sband/{GALAXY}/{GALAXY}_cs_o1_stokes_i.ms",
        infer=infer,
        npix=PIX_SIZE,
        res_in_arcsec=RES_IN_ARCSEC,
        device=DEVICE,
        num_major_cycles=NUM_MAJOR_CYCLES,
        vis_scale=VIS_SCALE,
        npz_name=NPZ_PATH,
        )
    model_image = result["model_image"][0,0].detach().cpu()
    #residual_images = np.stack(result["res_image"], axis=0)#result["res_image"][:][0,...].detach().cpu()
    residual_images = np.stack([r.detach().cpu().numpy() for r in result["res_image"]],axis=0,)
    model_plus_residual = model_image + torch.tensor(residual_images[-1,...], device=model_image.device)
    #save as fits
    header = wcs.to_header()

    # Optional, aber sinnvoll:
    header["BUNIT"] = "Jy/beam"   # oder Jy/pixel – je nach Training
    header["BTYPE"] = "Intensity"
    header["OBJECT"] = "Galaxy Cluster"
    header["EQUINOX"] = 2000.0
    header["RADESYS"] = "ICRS"

    hdu = fits.PrimaryHDU(data=np.flip(model_image.numpy(), axis=[0]), header=header)#np.flip(model_image.numpy().T, axis=[0,1])
    hdul = fits.HDUList([hdu])
    hdul.writeto(f"{SAVE_PATH}/{OBJ_NAME}/{OBJ_NAME}_network_reconstruction.fits", overwrite=True)

    hdu_mod_plus_res = fits.PrimaryHDU(data=np.flip(model_plus_residual.cpu().numpy(), axis=[0]), header=header)
    hdul_mod_plus_res = fits.HDUList([hdu_mod_plus_res])
    hdul_mod_plus_res.writeto(f"{SAVE_PATH}/{OBJ_NAME}/{OBJ_NAME}_network_reconstruction_plus_residual.fits", overwrite=True)

    header_res = wcs.to_header()
    header_res["NAXIS"]  = 3
    header_res["NAXIS1"] = residual_images.shape[-1]
    header_res["NAXIS2"] = residual_images.shape[-2]
    header_res["NAXIS3"] = residual_images.shape[0]

    header_res["CTYPE3"] = "CHANNEL"
    header_res["CUNIT3"] = ""
    header_res["CRPIX3"] = 1.0
    header_res["CRVAL3"] = 0.0
    header_res["CDELT3"] = 1.0

    residual_cube = np.flip(
    residual_images,
    axis=[1]   # gleiche Flip-Logik wie beim Model (Ny-Achse)
    )

    hdu_res = fits.PrimaryHDU(
        data=residual_cube,
        header=header_res
    )

    hdul_res = fits.HDUList([hdu_res])
    hdul_res.writeto(
        f"{SAVE_PATH}/{OBJ_NAME}/{OBJ_NAME}_network_residuals.fits",
        overwrite=True
    )

    print("Finished writing FITS file.")


