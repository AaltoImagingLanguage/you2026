########################################

# Generate RSA results for each parcel and each condition

import xarray as xr

import numpy as np
import time
import pickle
from config import (
    fname,
    subjects,
    time_len,
    time_interval,
    time_step,
    event_id,
    metric_rsa,
    metric_rdm,
)


import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--parcel_ind",
    type=int,
    default=40,
    help="index of parcel to analyze, e.g., 40 for vOT, 65 for ST",
)

args = parser.parse_args()

sti_types = list(event_id.keys())

tstart = time.monotonic()

roi_id = args.parcel_ind  # vOT, ST
data = xr.load_dataarray(fname.meg_tc_con(roi=roi_id)) # (subjects, conditions, vertices, time)

data = data.sel(time=slice(0, time_len))
times = data.time
times = data.time[::time_step]
times = times[
    times <= time_len - time_interval
]  # make sure the 100 ms time window extracted is within the data range


def generate_meg_rdms(subject):
    """Generate MEG RDMs for each time sample."""
    for t in times:
        data_sub = data.sel(subject=subject)
        yield mne_rsa.compute_rdm(
            data_sub.sel(time=slice(t, t + time_interval)).mean(dim="time").data,
            metric=metric_rdm,
        )


import mne_rsa

with open(
    fname.pcoder_reps,
    "rb",
) as f:
    rep_dict = pickle.load(f)

rsa_results_allp = []
for p in rep_dict:
    rdms = []
    data_p = np.array([rep_dict[p][sti] for sti in sti_types])
    data_p = data_p.mean(1)  # average across stimuli within each condition
    for it in range(2):  # two states: w/ feedback, w/o feedback

        reps_array = data_p[:, it, :]
        rdm = mne_rsa.compute_rdm(reps_array, metric=metric_rdm)
        rdms.extend([rdm])

    # compute RAS for each pcoder
    rsa_results_all = []
    for subject in subjects:

        rsa_results = mne_rsa.rsa(
            generate_meg_rdms(subject),
            rdms,
            metric=metric_rsa,  # spearman,pearson, kendall-tau-a, regression
            verbose=True,
            n_data_rdms=len(times),
            n_jobs=1,
        )
        rsa_results_all.append(rsa_results)
    rsa_results_allp.append(rsa_results_all)
rsa_results_allp = np.array(rsa_results_allp)  # (23,51,2)

# Convert rsa_results_allp to xarray DataArray
rsa_results_allp = np.array(rsa_results_allp)  # Shape: (3, 23, 51, 2)

# Create the xarray DataArray with proper dimensions and coordinates
rsa_xr = xr.DataArray(
    rsa_results_allp,
    dims=["pcoder", "subject", "time", "feedback"],
    coords={
        "pcoder": range(3),  # P-coders 0, 1, 2
        "subject": subjects,  # Your subject list
        "time": times.values,  # MEG time points
        "feedback": ["w/o", "w/"],  # Model timesteps [0, 23]
    },
    attrs={
        "description": f"RSA result for model and parcel {roi_id}",
        "metric_rsa": metric_rsa,
    },
)

# Save to NetCDF file

rsa_xr.to_netcdf(fname.rsa_tc(roi=roi_id))

print(f"RSA results saved to: {fname.rsa_tc(roi=roi_id)}")
print((time.monotonic() - tstart) / 60)
