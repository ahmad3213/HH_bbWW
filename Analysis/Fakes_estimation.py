import ROOT
import sys, os

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.HistHelper import *
from FLAF.Common.Utilities import *

# ============================================================
# OLD FAKE FACTOR METHOD (UNCHANGED)
# ============================================================


def Fakes_Estimation_BBWW(
    histograms, all_samples_list, channel, category, uncName, scale, data_process_name
):

    key_Anti = ((channel, "AR_AntiTightId", category), (uncName, scale))

    hist = histograms[data_process_name][key_Anti].Clone()
    hist.SetDirectory(0)

    for sample in all_samples_list:

        if sample == data_process_name or sample == "QCD":
            continue

        if sample not in histograms:
            continue

        if key_Anti not in histograms[sample]:
            continue

        hist.Add(histograms[sample][key_Anti], -1)

    hist_fakes_Central = hist.Clone()
    hist_fakes_Up = hist_fakes_Central.Clone()
    hist_fakes_Down = hist_fakes_Central.Clone()

    n_yield = hist_fakes_Central.Integral(0, hist_fakes_Central.GetNbinsX() + 1)

    return hist_fakes_Central, hist_fakes_Up, hist_fakes_Down, n_yield, 0.0


def AddFakesInHistDict_BBWW(
    var,
    all_histograms,
    channels,
    categories,
    uncName,
    all_samples_list,
    scales,
    data_process_name=None,
):

    if "Fakes" not in all_histograms:
        all_histograms["Fakes"] = {}

    for channel in channels:
        for cat in categories:

            if cat == "boosted":
                continue

            for scale in scales + ["Central"]:

                if uncName == "Central" and scale != "Central":
                    continue
                if uncName != "Central" and scale == "Central":
                    continue

                key = ((channel, "ZVeto_OS_Iso", cat), (uncName, scale))

                hist, _, _, yield_est, _ = Fakes_Estimation_BBWW(
                    all_histograms,
                    all_samples_list,
                    channel,
                    cat,
                    uncName,
                    scale,
                    data_process_name,
                )

                all_histograms["Fakes"][key] = hist

                if uncName == "Central":
                    print(f"[FF] {channel} {cat} {var} yield={yield_est}")


def _get_data_minus_mc(histograms, backgrounds_list, key, data_process_name):
    if data_process_name not in histograms:
        raise KeyError(f"Missing data process {data_process_name}")
    if key not in histograms[data_process_name]:
        raise KeyError(f"Missing key {key} in data")
    hist = histograms[data_process_name][key].Clone()
    hist.SetDirectory(0)
    skip_background_samples = {"QCD_PT"}
    if not backgrounds_list:
        raise ValueError("Empty background list passed to fake estimation")
    for sample in backgrounds_list:

        if sample in skip_background_samples:
            continue

        if sample == data_process_name:
            continue

        if sample not in histograms:
            print(f"[WARN] sample {sample} missing in histograms")
            continue

        if key not in histograms[sample]:
            continue

        hist.Add(histograms[sample][key], -1)

    return hist


# ============================================================
# SAFE FLATTENING (OPTIONAL)
# ============================================================


def flatten_2d_to_1d(hist2d, combined_bins, name="flat"):

    if combined_bins is None:
        raise ValueError("combined_bins is None but flattening was requested")

    total_bins = sum(len(b["x_bins"]) - 1 for b in combined_bins)
    h1 = ROOT.TH1D(name, name, total_bins, 0, total_bins)

    bin_counter = 1

    for block in combined_bins:

        y_low, y_high = block["y_bin"]
        x_bins = block["x_bins"]

        y_bin_min = hist2d.GetYaxis().FindBin(y_low)
        y_bin_max = hist2d.GetYaxis().FindBin(y_high - 1e-6)

        for i in range(len(x_bins) - 1):

            x_low = x_bins[i]
            x_high = x_bins[i + 1]

            x_bin_min = hist2d.GetXaxis().FindBin(x_low)
            x_bin_max = hist2d.GetXaxis().FindBin(x_high - 1e-6)

            content = 0.0

            for ix in range(x_bin_min, x_bin_max + 1):
                for iy in range(y_bin_min, y_bin_max + 1):
                    content += hist2d.GetBinContent(ix, iy)

            h1.SetBinContent(bin_counter, content)
            h1.SetBinError(bin_counter, 0.0)

            bin_counter += 1

    return h1


# ============================================================
# MAIN FAKE BUILDER (AUTO 1D / 2D MODE)
# ============================================================


def AddFakesInHistDict_BBWW_TransferFactor(
    var,
    all_histograms,
    channels,
    categories,
    uncName,
    backgrounds,
    scales,
    data_process_name,
):
    if "Fakes" not in all_histograms:
        all_histograms["Fakes"] = {}

    for channel in channels:
        for cat in categories:

            if cat == "boosted":
                continue

            for scale in scales + ["Central"]:

                if uncName == "Central" and scale != "Central":
                    continue
                if uncName != "Central" and scale == "Central":
                    continue

                # Fakes = data -MC for Anti tight region only
                if channel in ("e", "eE", "eMu"):
                    anti_key = ((channel, "AR_AntiTightId", cat), (uncName, scale))
                if channel in ("mu", "muMu"):
                    anti_key = ((channel, "OS_AntiIso", cat), (uncName, scale))
                hist_anti = _get_data_minus_mc(
                    all_histograms, backgrounds, anti_key, data_process_name
                )
                # Compute Data - MC in the tight region as well
                signal_key = ((channel, "OS_Iso", cat), (uncName, scale))
                hist_signal = _get_data_minus_mc(
                    all_histograms, backgrounds, signal_key, data_process_name
                )

                hist_signal = hist_signal.Clone(f"Fake_{channel}_{cat}")
                hist_signal.SetDirectory(0)

                hist_anti = hist_anti.Clone(f"Fake_{channel}_{cat}_anti")
                hist_anti.SetDirectory(0)

                # saving Fakes in both tight and AntiTight
                all_histograms["Fakes"][anti_key] = hist_anti
                all_histograms["Fakes"][signal_key] = hist_signal
