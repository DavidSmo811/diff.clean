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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#str(gpu_id) Ich musste hier 0 setzen, sonst habe ich bei der Inferenz immer einen CUDA-Fehler bekommen. Non-uniform Points usw., das ging hiermit zu korrigieren

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



def read_ms(msname, field_id=0, device="cpu"):
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
    
    # # Teile Channels in Batches
    # channels_per_batch = nchan // num_channel_batches
    # if nchan % num_channel_batches != 0:
    #     channels_per_batch += 1
    
    # print(f"\nChannel Batching:")
    # print(f"  Total channels: {nchan}")
    # print(f"  Number of batches: {num_channel_batches}")
    # print(f"  Channels per batch: ~{channels_per_batch}")
    
    vis_list = []
    uvw_list = []
    nu_list = []
    
    
    # Channel-Range für diesen Batch
    #ch_start = batch_idx * channels_per_batch
    #ch_end = min((batch_idx + 1) * channels_per_batch, nchan)
    
    #if ch_start >= nchan:
    #    break
    
    #print(f"\n  Processing batch {batch_idx+1}/{num_channel_batches}: channels {ch_start}-{ch_end-1}")
    
    # Slice für diesen Batch
    nchan_range = range(nchan)
    for i in nchan_range:
        print("Min uvw in m",np.min(uvw_m), "Max uvw in m", np.max(uvw_m))
        vis_chan = vis_np[:, i:i+1, :]      # (nrow, n_ch_batch, ncorr)
        flag_chan = flag_np[:, i:i+1, :]
        freq_chan = freq[i]
        
        if use_weights:
            weights_batch = weights_np[:, i:i+1, :]
        
        # Mittlere Frequenz dieses Batches
        #nu_batch = float(np.mean(freq_batch))
        lambda_batch = c / freq_chan
        
        # UVW für diese mittlere Frequenz
        uvw_batch = uvw_m / lambda_batch
        
        # Zu Torch
        vis_t = torch.as_tensor(vis_chan, dtype=torch.complex64, device=device)
        flag_t = torch.as_tensor(flag_chan, dtype=torch.bool, device=device)
        
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
            vis_t = vis_t[:,0,:]#torch.mean(vis_t, dim=1)
        
        # Stokes I
        vis_stokes = 0.5 * (vis_t[:, 0] + vis_t[:, 3])  # (nrow,)

        mask = vis_stokes != 0

        # Torch-Tensor filtern
        vis_stokes = vis_stokes[mask]
        if vis_stokes.numel() == 0:
            print(f"⚠️ Skip channel {freq_chan/1e9:.3f} GHz – no valid vis")
            continue

        # UVW erst zu Torch, dann filtern
        uvw_t = torch.as_tensor(uvw_batch, dtype=torch.float32, device=device)
        uvw_t = uvw_t[mask, :]

        # Zu Liste hinzufügen
        vis_list.append(vis_stokes.unsqueeze(0))  # (1, nrow)
        
        uvw_list.append(uvw_t.T.unsqueeze(0))  # (1, 3, nrow)
        
        nu_list.append(freq_chan)
        
        print(f"    Frequency: {freq_chan/1e9:.3f} GHz")
        print(f"    Visibilities: {vis_stokes.shape[0]}")
    
    vis= torch.cat(vis_list, dim=1)  # (n_batches, nrow)
    uvw= torch.cat(uvw_list, dim=2)  # (n_batches, 3, nrow)
    
    return vis, uvw, nu_list, ra0, dec0


def make_lmn_grid(npix, res_in_arcsec, dec=0, device="cpu"):
    """
    Erstellt ein (l,m,n) Grid basierend auf pyvisgen approach
    
    Parameters:
    -----------
    npix : int
        Anzahl der Pixel pro Achse
    res_in_arcsec : float
        Pixelgröße in arcsec
    dec : float
        Deklination in Radiant (für Projektionskorrektur)
    
    Returns:
    --------
    lmn : np.ndarray
        Array mit Shape (3, npix, npix) mit [L, M, N-1]
    """
    # FOV in Radiant
    fov_rad = (npix * res_in_arcsec / 3600.0) * np.pi / 180.0#Eigentlich npix statt npix/2
    
    # Erstelle lineare Achsen von -fov/2 bis +fov/2
    axis = np.linspace(-fov_rad / 2, fov_rad / 2, npix)
    
    # Meshgrid für L und M
    L, M = np.meshgrid(axis, axis, indexing='xy')

    # N berechnen: n = sqrt(1 - l^2 - m^2)  
    # Für den w-term wird oft (n-1) verwendet
    lm2 = L**2 + M**2
    N = np.sqrt(np.maximum(0.0, 1.0 - lm2))#np.ones_like(L)#np.sqrt(np.maximum(0.0, 1.0 - lm2)) - 1.0  # -1 für w-term
    
    # Stack zu (3, npix, npix)
    lmn = np.stack([L, M, N], axis=0)
    print("L,M,N shape:", lmn.shape)
    
    lmn_torch = torch.tensor(lmn, dtype=torch.float32, device=device)
    
    # Reshape: (3, npix, npix) -> (3, npix*npix)
    lmn_flat = lmn_torch.reshape(3, -1)
    
    # Batch dimension: (3, npix*npix) -> (1, 3, npix*npix)
    lmn_batched = lmn_flat.unsqueeze(0)
    
    print(f"\nLMN prepared for inference:")
    print(f"  Shape: {lmn_batched.shape}")
    print(f"  L: min={lmn_batched[0, 0].min():.6f}, max={lmn_batched[0, 0].max():.6f}, unique={len(torch.unique(lmn_batched[0, 0]))}")
    print(f"  M: min={lmn_batched[0, 1].min():.6f}, max={lmn_batched[0, 1].max():.6f}, unique={len(torch.unique(lmn_batched[0, 1]))}")
    print(f"  N: min={lmn_batched[0, 2].min():.6f}, max={lmn_batched[0, 2].max():.6f}, unique={len(torch.unique(lmn_batched[0, 2]))}")
    
    return lmn_batched


def make_lmn_grid_pyvis(npix, res_in_arcsec, dec=0, device="cpu"):
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

def run_inference_from_ms(
    msname,
    infer,
    npix=512,
    res_in_arcsec=1.0,
    device="cpu",
    num_major_cycles=7,
    vis_scale=1,
    npz_name=None,
):
    if npz_name is not None:
        try:
            print("Loading preprocessed data from NPZ...")
            saved = np.load(npz_name)
            vis  = torch.tensor(saved["vis"], device=device)
            uvw  = torch.tensor(saved["uvw"], device=device)
            nu   = saved["nu"].item()
            ra0  = saved["ra0"].item()
            dec0 = saved["dec0"].item()
            print("Loaded preprocessed data.")
        except Exception:
            print("NPZ not found, reading MS...")
            vis, uvw, nu, ra0, dec0 = read_ms(msname, device=device)
            npz_path = Path(npz_name)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz_name,
                     vis=vis.cpu().numpy(),
                     uvw=uvw.cpu().numpy(),
                     nu=nu, ra0=ra0, dec0=dec0,)
    else:
        print("No NPZ specified, reading MS...")
        vis, uvw, nu, ra0, dec0 = read_ms(msname, device=device)
    wcs = make_wcs(npix, res_in_arcsec, ra0, dec0)
    #uvw = uvw_to_lambda(uvw, nu)

    print("Building LMN grid...")
    lmn = make_lmn_grid_pyvis(npix, res_in_arcsec,device=device, dec=np.degrees(dec0))
    
    # Torch
    vis = torch.tensor(vis, dtype=torch.complex64, device=device)[None, :]*vis_scale
    uvw = torch.tensor(uvw, dtype=torch.float32, device=device)#[None, :, :].T.unsqueeze(0)
    #lmn = torch.flatten(torch.tensor(lmn, dtype=torch.float32, device=device)[None, ...],start_dim=2, end_dim=3)
    print(lmn.shape)
    print(vis.shape)
    print(lmn[:,:,:10])

    print("Running inference...")
    result = infer.run_major_minor(
        vis=vis,
        uvw=uvw,
        lmn=lmn,
        num_major_cycles=num_major_cycles,
    )

    return result, wcs

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
    TRAINED_MAJOR_CYCLES = 7
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
        num_conditions=TRAINED_MAJOR_CYCLES + 1,
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
    residual_images = np.stack([r for r in result["res_image"]],axis=0,)
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


