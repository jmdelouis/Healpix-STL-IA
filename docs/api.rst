API reference
=============

Generated from the docstrings of the code at every build.

Network
-------

.. autosummary::
   :toctree: generated

   stlia.unet.UNet
   stlia.unet.load
   stlia.unet.SphereLevel
   stlia.unet.SphereConv
   stlia.unet.SphereBlock
   stlia.unet.PixelNorm

Losses and statistics
---------------------

.. autosummary::
   :toctree: generated

   stlia.stl_torch.DenoiseLoss
   stlia.stl_torch.Stats
   stlia.stl_torch.StatsFunction
   stlia.stl_torch.as_real
   stlia.stl_torch.native_pixels

Spherical harmonics and spectra
-------------------------------

.. autosummary::
   :toctree: generated

   stlia.sht.HEALPixSHT
   stlia.spectra.BinnedSpectra
   stlia.spectra.SpectrumLoss
   stlia.spectra.galactic_mask
   stlia.spectra.bin_edges

Disc patches
------------

.. autosummary::
   :toctree: generated

   stlia.regions.Patches
   stlia.regions.Level
   stlia.regions.disc_ids
   stlia.regions.pixel_vectors
   stlia.regions.stencil
   stlia.regions.bound_stencil

Training
--------

.. autosummary::
   :toctree: generated

   stlia.train_denoise
   stlia.train_patches

Geometry
--------

.. autosummary::
   :toctree: generated

   stlia.stl_torch_geometry
   stlia.healpix_faces
