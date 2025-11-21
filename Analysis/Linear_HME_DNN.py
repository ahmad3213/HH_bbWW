from __future__ import annotations


class LinHMEDNN:
    def __init__(self, cfg, payload_name):

        self.cfg = cfg
        self.payload_name = payload_name

        self.hme_bins_dict = {
            "DL": {
                "300": [-10, 250, 275, 325, 400, 10_000],
                "400": [-10, 300, 350, 450, 500, 10_000],
                "500": [-10, 400, 450, 550, 600, 10_000],
                "550": [-10, 400, 475, 625, 700, 10_000],
                "600": [-10, 500, 550, 650, 700, 10_000],
                "650": [-10, 500, 550, 700, 800, 10_000],
                "700": [-10, 550, 625, 750, 800, 10_000],
                "800": [-10, 650, 700, 850, 950, 10_000],
                "900": [-10, 700, 800, 950, 1050, 10_000],
                "1000": [-10, 750, 875, 1075, 1150, 10_000],
            },
            "SL": {  # SL copied from DL for now
                "300": [-10, 250, 275, 325, 400, 10_000],
                "400": [-10, 300, 350, 450, 500, 10_000],
                "500": [-10, 400, 450, 550, 600, 10_000],
                "550": [-10, 400, 475, 625, 700, 10_000],
                "600": [-10, 500, 550, 650, 700, 10_000],
                "650": [-10, 500, 550, 700, 800, 10_000],
                "700": [-10, 550, 625, 750, 800, 10_000],
                "800": [-10, 650, 700, 850, 950, 10_000],
                "900": [-10, 700, 800, 950, 1050, 10_000],
                "1000": [-10, 750, 875, 1075, 1150, 10_000],
            },
        }

        # self.hme_bins = [
        #     -10,
        #     225,
        #     275,
        #     325,
        #     375,
        #     425,
        #     525,
        #     675,
        #     850,
        #     1250,
        #     1750,
        #     2250,
        #     3250,
        #     10000,
        # ]

        self.channel = cfg["channel"]

        self.hme_bins = self.hme_bins_dict[self.channel]

    def run(self, dfw):
        for col in self.cfg["columns"]:
            mass = col.split("_")[-1][1:]
            HME_var = (
                "DoubleLep_DeepHME_mass"
                if self.channel == "DL"
                else "SingleLep_DeepHME_mass"
            )
            DNN_var = (
                f"DNNParametric_DL_NoHME_M{mass}_Signal"
                if self.channel == "DL"
                else f"DNNParametric_SL_NoHME_M{mass}_Signal"
            )

            hme_bins_string = ",".join(map(str, self.hme_bins[str(mass)]))
            n_bins = len(self.hme_bins[str(mass)]) - 1

            dfw.Define(
                f"linear_hme_dnn_m{mass}",
                f"""
            static const Double_t bins[{n_bins+1}] = {{{hme_bins_string}}};
            static const TAxis axis({n_bins}, bins);
            return axis.FindFixBin({HME_var}) - 1 + {DNN_var};
            """,
            )
        for col in self.cfg["columns"]:
            dfw.DefineAndAppend(f"{self.payload_name}_{col}", f"return {col};")
        return dfw
