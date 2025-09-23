import ROOT
import yaml
import os
import sys
import json


class LHE_PT_NJetStitcher:
    """
    Processor for DY stitching using LHE_Vpt and LHE_NpNLO selections.
    It reads the stitching config (bin selections)
    and computes the event yields per bin.
    """

    def __init__(self, global_params, config_path=None, verbose=0):
        self.global_params = global_params
        self.enabled = global_params.get("DY_stitched_enable", True)
        self.config_path = config_path
        self.verbose = verbose
        self.bins = []

        if not self.enabled:
            print(
                "[LHE_PT_NJetStitcher] Stitching disabled via global_params.",
                file=sys.stderr,
            )
            return

        if not config_path or not os.path.exists(config_path):
            print(
                f"[LHE_PT_NJetStitcher] ERROR: Config file not found: {config_path}",
                file=sys.stderr,
            )
            return

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.bins = cfg.get("stitched_binning", [])
        if not self.bins:
            print(
                "[LHE_PT_NJetStitcher] WARNING: No stitched_binning defined in config.",
                file=sys.stderr,
            )

        print(
            f"[LHE_PT_NJetStitcher] Loaded {len(self.bins)} stitching bins from {config_path}.",
            file=sys.stderr,
        )

    # ----------------------------------------------------------------
    # HOOK: onAnaCacheTask
    # ----------------------------------------------------------------
    def onAnaCacheTask(self, df):
        """
        Compute the number of events in each stitched bin, weighted by denominator.
        Returns stitching summary to merge into anaCache.
        """
        if not self.enabled or not self.bins:
            print(
                "[LHE_PT_NJetStitcher] Skipping stitching computation.", file=sys.stderr
            )
            return None

        evts_per_bin = {}
        total_sumw = 0.0
        total_n_events = 0.0
        if self.verbose:
            print(f"[DEBUG] Type of df: {type(df)}", file=sys.stderr)
        if self.verbose:
            print(
                "[LHE_PT_NJetStitcher] Starting stitching computation...",
                file=sys.stderr,
            )

        for bin_cfg in self.bins:
            name = bin_cfg["name"]
            selection = bin_cfg["bin_selection"]
            if self.verbose:
                print(
                    f"[LHE_PT_NJetStitcher] Processing bin '{name}': selection = {selection}",
                    file=sys.stderr,
                )
            # Apply selection and sum denominator weights
            df_sel = df.Filter(selection)
            n_events = df_sel.Count().GetValue()
            sum_w = df_sel.Sum("weight_denom_Central").GetValue()
            evts_per_bin[name] = {
                "selection": selection,
                "sum_weights": sum_w,
                "n_events": n_events,
            }
            total_sumw += sum_w
            total_n_events += n_events
            if self.verbose:
                print(
                    f"[LHE_PT_NJetStitcher] Bin {name}: evts_per_bin={n_events}",
                    file=sys.stderr,
                )
                print(
                    f"[LHE_PT_NJetStitcher] Bin {name}: sum_weights={sum_w:.6f}",
                    file=sys.stderr,
                )

        # Return to be merged into anaCache
        results = {
            "DY_stitching": {
                "bins": evts_per_bin,
                "total_denom_sum": total_sumw,
                "total_n_events": total_n_events,
            }
        }
        if self.verbose:
            print(
                f"[DEBUG] Returning result from LHE_PT_NJetStitcher: {type(results)}",
                file=sys.stderr,
            )
        return results

    def mergeAnaCache(self, anaCache, newResults):
        """Merge newResults into anaCache"""
        for key, value in newResults.items():
            if key in anaCache:
                # Merge dicts recursively
                if isinstance(anaCache[key], dict) and isinstance(value, dict):
                    self.mergeAnaCache(anaCache[key], value)
                # Sum numbers
                elif isinstance(anaCache[key], (int, float)) and isinstance(
                    value, (int, float)
                ):
                    anaCache[key] += value
                # Overwrite otherwise
                else:
                    anaCache[key] = value
            else:
                anaCache[key] = value
        return anaCache

    # ----------------------------------------------------------------
    # Optional hooks (placeholders for future)
    # ----------------------------------------------------------------
    def onAnaTupleProd(self, df, global_params, samples, sample, anaCache=None):
        """
        save an additional fraction weight which can convert classical (non stitched) to stitched event weight
        """
        luminosity = global_params["luminosity"]
        xsFile = global_params["crossSectionsFile"]
        xsFilePath = os.path.join(os.environ["ANALYSIS_PATH"], xsFile)
        with open(xsFilePath, "r") as xs_file:
            xs_dict = yaml.safe_load(xs_file)
        xs_name = samples[sample]["crossSection"]
        xs_inclusive = xs_dict[xs_name]["crossSec"]
        DY_stitching_extra_branches = []

        bin_expr = ""
        for i, bin_cfg in enumerate(self.bins, start=1):
            selection = bin_cfg["bin_selection"]
            bin_expr += f"({selection}) ? {i} : "
        bin_expr += "0"  # default bin
        df = df.Define("DY_stitch_bin", bin_expr)
        DY_stitching_extra_branches.append("DY_stitch_bin")

        df = df.Define("DY_MC_stitch_LHE_Vpt", "LHE_Vpt")
        DY_stitching_extra_branches.append("DY_MC_stitch_LHE_Vpt")
        df = df.Define("DY_MC_stitch_NpNLO", "LHE_NpNLO")
        DY_stitching_extra_branches.append("DY_MC_stitch_NpNLO")
        # Stitching normalization
        xs_total = self.safe_eval(xs_inclusive)
        lumi = self.safe_eval(luminosity)
        total_denom_sum = self.safe_eval(anaCache["DY_stitching"]["total_denom_sum"])
        total_n_events = self.safe_eval(anaCache["DY_stitching"]["total_n_events"])

        # Cross section per bin (fractional)
        per_bin_xs = {
            k: (
                xs_total * (v["n_events"] / total_n_events)
                if total_n_events > 0
                else 0.0
            )
            for k, v in anaCache["DY_stitching"]["bins"].items()
        }

        xs_expr = ""
        denom_expr = ""
        fraction_expr = ""

        for i, (bin_name, bininfo) in enumerate(
            anaCache["DY_stitching"]["bins"].items(), start=1
        ):
            xs_bin = per_bin_xs[bin_name]
            sum_w_bin = bininfo["sum_weights"]
            n_evt_bin = bininfo["n_events"]

            # cross-section expression
            xs_expr += f"(DY_stitch_bin == {i}) ? {xs_bin} : "

            # denominator expression
            denom_expr += f"(DY_stitch_bin == {i}) ? {sum_w_bin} : "

            # fraction expression
            if n_evt_bin > 0 and total_n_events > 0 and sum_w_bin > 0:
                frac = (total_denom_sum * (n_evt_bin / total_n_events)) / sum_w_bin
            else:
                frac = 1.0
            fraction_expr += f"(DY_stitch_bin == {i}) ? {frac} : "

        # Add default fallbacks
        xs_expr += "1.0"
        denom_expr += "1.0"
        fraction_expr += "1.0"
        df = df.Define("DY_MC_stitch_weight_frac", f"(float)({fraction_expr})")
        DY_stitching_extra_branches.append("DY_MC_stitch_weight_frac")

        return df, DY_stitching_extra_branches

    def onHistTupleProd(self, df, weight_var_name):
        """
        Modifies weight_var_name in place to include DY_MC_stitch_weight_frac if present,
        but always keeps the same column name (e.g. 'weight_Central').
        """

        # If the stitching column exists, multiply in place
        if df.HasColumn("DY_MC_stitch_weight_frac"):
            expr = f"{weight_var_name} * DY_MC_stitch_weight_frac"

            if weight_var_name in df.GetDefinedColumnNames():
                print(
                    f"[LHE_PT_NJetStitcher] Redefining '{weight_var_name}' to include stitching weight."
                )
                df = df.Redefine(weight_var_name, expr)
            else:
                print(
                    f"[LHE_PT_NJetStitcher] Defining '{weight_var_name}' with stitching weight."
                )
                df = df.Define(weight_var_name, expr)

        else:
            # No stitching column — keep the original weight as-is
            print(
                f"[LHE_PT_NJetStitcher] Info: 'DY_MC_stitch_weight_frac' not found, keeping '{weight_var_name}' unchanged."
            )

        # Always return the same column name to maintain downstream compatibility
        return df, weight_var_name

    def safe_eval(self, expr):
        if isinstance(expr, str):
            return float(eval(expr))  # only safe for simple arithmetic expressions
        else:
            return float(expr)
