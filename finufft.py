import warnings
from functools import partial
from math import pi

import cufinufft
import cupy as cp
import torch
from operator import itemgetter


class CupyFinufft:
    """Wraper to use Finufft Type 3d3 for radio interferometry data."""

    def __init__(
        self,
        image_size,
        fov_arcsec,
        eps=1e-12,
    ):
        """Wraper to use Finufft Type 3d3 for radio interferometry data."""
        self.px_size = ((fov_arcsec / 3600) * pi / 180) / image_size
        self.px_scaling = image_size**2
        self.image_size = image_size

        self.ft = partial(cufinufft.nufft3d3, isign=-1, eps=eps)
        self.ift = partial(cufinufft.nufft3d3, isign=+1, eps=eps)
    @profile
    def _compute_visibility_weights_and_indices(
        self,
        u_coords,
        v_coords,
        w_coords,
        image_size=None,
    ):
        """
        Compute visibility weights and bin indices for each visibility sample.

        This method maps each visibility to the corresponding image pixel and
        returns both the bin count for that pixel and the indices needed for
        normalization.

        Parameters
        ----------
        u_coords : torch.Tensor
            U coordinates of visibility samples
        v_coords : torch.Tensor
            V coordinates of visibility samples
        w_coords : torch.Tensor
            W coordinates of visibility samples (not used for 2D binning)
        image_size : int, optional
            Size of the output image. If None, uses self.image_size

        Returns
        -------
        visibility_weights : torch.Tensor
            1D tensor of length num_visibilities containing the bin count
            for each visibility sample (i.e., how many visibilities map to
            the same pixel as this visibility)
        """
        if image_size is None:
            image_size = self.image_size

        # Convert UV coordinates to pixel indices
        u_pixels = (u_coords / (2 * pi / image_size)).astype(cp.int64)
        v_pixels = (v_coords / (2 * pi / image_size)).astype(cp.int64)

        # Clip to valid image bounds
        u_pixels = cp.clip(u_pixels, -image_size // 2, image_size // 2 - 1)
        v_pixels = cp.clip(v_pixels, -image_size // 2, image_size // 2 - 1)

        # Shift to positive indices (center at image_size // 2)
        u_pixels = u_pixels + image_size // 2
        v_pixels = v_pixels + image_size // 2

        # Ensure indices are within bounds
        valid_mask = (
            (u_pixels >= 0)
            & (u_pixels < image_size)
            & (v_pixels >= 0)
            & (v_pixels < image_size)
        )

        # Convert 2D indices to 1D for bincount
        linear_indices = v_pixels * image_size + u_pixels

        # Compute histogram of all visibilities
        histogram_flat = cp.bincount(
            linear_indices, minlength=image_size * image_size
        ).astype(cp.float64)

        # Map each visibility to its bin count
        # This tells us how many visibilities fall into the same bin as each visibility
        visibility_weights = histogram_flat[linear_indices]

        # Handle invalid entries (outside bounds)
        visibility_weights[~valid_mask] = 1.0

        return visibility_weights
    @profile
    def nufft(
        self,
        sky_values,
        l_coords,
        m_coords,
        n_coords,
        u_coords,
        v_coords,
        w_coords,
        return_torch=False,
    ):
        """Calculate the fast Fourier transform with non-uniform source
        and non-uniform target coordinates.
        """
        # Sky coordinates (Image domain - lmn coordinates)
        source_l = cp.asarray((l_coords / self.px_size), dtype=cp.float64)
        source_m = cp.asarray((m_coords / self.px_size), dtype=cp.float64)
        source_n = cp.asarray(((n_coords - 1) / self.px_size), dtype=cp.float64)

        # Antenna coordinates (Fourier Domain - uvw coordinates)
        target_u = cp.asarray(
            2 * pi * (u_coords.flatten() * self.px_size),
            dtype=cp.float64,
        )
        target_v = cp.asarray(
            2 * pi * (v_coords.flatten() * self.px_size),
            dtype=cp.float64,
        )
        target_w = cp.asarray(
            2 * pi * (w_coords.flatten() * self.px_size),
            dtype=cp.float64,
        )

        outside_bounds = cp.array(
            [
                (target_u.get() <= -pi) | (target_u.get() > pi),
                (target_v.get() <= -pi) | (target_v.get() > pi),
                (target_w.get() <= -pi) | (target_w.get() > pi),
            ]
        )
        coord_outside = cp.where(cp.any(outside_bounds, axis=1))[0]
        uvw_map = {
            0: "u",
            1: "v",
            2: "w",
        }
        if outside_bounds.any():
            warnings.warn(
                f"Some of the {', '.join(itemgetter(*coord_outside)(uvw_map))} coordinates "
                "lie outside the constructed image. This can lead to cufinufft errors."
            )

        # Values at source position (Source intensities)
        c_values = cp.asarray(sky_values.flatten(), dtype=cp.complex128)

        result = self.ft(
            source_l,
            source_m,
            source_n,
            c_values,
            target_u,
            target_v,
            target_w,
        )

        if return_torch:
            visibilities = torch.as_tensor(result, device="cuda")
        else:
            visibilities = result.get()

        return visibilities
    @profile
    def inufft(
        self,
        visibilities,
        l_coords,
        m_coords,
        n_coords,
        u_coords,
        v_coords,
        w_coords,
        return_torch=False,
    ):
        """Calculate the inverse fast Fourier transform with non-uniform source
        and non-uniform target coordinates.
        """
        # Antenna coordinates (Fourier Domain - uvw coordinates)
        source_u = cp.asarray(
            2 * pi * (u_coords.flatten() * self.px_size), dtype=cp.float64
        )
        source_v = cp.asarray(
            2 * pi * (v_coords.flatten() * self.px_size), dtype=cp.float64
        )
        source_w = cp.asarray(
            2 * pi * (w_coords.flatten() * self.px_size), dtype=cp.float64
        )

        # Compute visibility weights: for each visibility, how many other
        # visibilities fall into the same UV bin?
        visibility_weights = self._compute_visibility_weights_and_indices(
            source_u,
            source_v,
            source_w,
            image_size=self.image_size,
        )

        outside_bounds = cp.array(
            [
                (source_u.get() <= -pi) | (source_u.get() > pi),
                (source_v.get() <= -pi) | (source_v.get() > pi),
                (source_w.get() <= -pi) | (source_w.get() > pi),
            ]
        )
        coord_outside = cp.where(cp.any(outside_bounds, axis=1))[0]
        uvw_map = {
            0: "u",
            1: "v",
            2: "w",
        }
        if outside_bounds.any():
            warnings.warn(
                f"Some of the {', '.join(itemgetter(*coord_outside)(uvw_map))} coordinates "
                "lie outside the constructed image. This can lead to cufinufft errors."
            )

        # Fourier coeficients at antenna positions (Visibilities)
        c_values = cp.asarray(visibilities.flatten(), dtype=cp.complex128)

        # Normalize visibility values by dividing by their bin counts
        # This means visibilities that fall into the same bin are averaged
        c_values_normalized = c_values / visibility_weights

        # Sky coordinates (Image domain - lmn coordinates)
        target_l = cp.asarray((l_coords / self.px_size), dtype=cp.float64)
        target_m = cp.asarray((m_coords / self.px_size), dtype=cp.float64)
        target_n = cp.asarray(((n_coords - 1) / self.px_size), dtype=cp.float64)

        result = (
            self.ift(
                source_u,
                source_v,
                source_w,
                c_values_normalized,
                target_l,
                target_m,
                target_n,
            )
            / self.px_scaling
        )

        if return_torch:
            sky_intensities = torch.as_tensor(result, device="cuda")
        else:
            sky_intensities = result.get()

        return sky_intensities