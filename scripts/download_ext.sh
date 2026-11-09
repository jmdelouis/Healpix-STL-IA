#!/bin/bash
# External maps of the figures, into $R/data/ext (about 1.6 GB):
#   the PR3 GNILC 353 GHz dust map (variable resolution, IRSA mirror of the Planck Legacy Archive) and
#   the BICEP2/Keck BK14 field mask in galactic coordinates (bicepkeck.org).
#
#   bash scripts/download_ext.sh
#
# Interrupted downloads resume (curl -C -).
: "${R:?source scripts/env.sh (or a site file) first}"
E=$R/data/ext
mkdir -p $E
cd $E
set -e
G=COM_CompMap_IQU-thermaldust-gnilc-varres_2048_R3.00.fits
until curl -sSf -C - -m 1800 -o $G \
    https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/foregrounds/$G; do
    echo "retrying $G"
    sleep 30
done
curl -sSf -m 120 -o bk14_mask_gal_n0512.fits http://bicepkeck.org/BK14_datarelease/bk14_mask_gal_n0512.fits
ls -la $E
