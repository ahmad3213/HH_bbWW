from FLAF.Common.Utilities import *

channels = [
    "muMu",
    "eMu",
    "eE",
    "mu",
    "e",
]  # in order of importance during the channel selection
leg_names = ["Electron", "Muon"]


def getChannelLegs(channel):
    ch_str = channel.lower()
    legs = []
    while len(ch_str) > 0:
        name_idx = None
        obj_name = None
        for idx, obj in enumerate(["e", "mu"]):
            if ch_str.startswith(obj):
                name_idx = idx
                obj_name = obj
                break
        if name_idx is None:
            raise RuntimeError(f"Invalid channel name {channel}")
        legs.append(leg_names[name_idx])
        ch_str = ch_str[len(obj_name) :]
    return legs


def RecoHWWCandidateSelection(df):
    df = df.Define(
        "Electron_sel",
        """
        v_ops::pt(Electron_p4) > 10 && abs(v_ops::eta(Electron_p4)) < 2.5 && abs(Electron_dz) < 0.1 && abs(Electron_dxy) < 0.05 && Electron_sip3d <= 8 && Electron_mvaIso_WP90""",
    )
    # Lower the muon pt threshold to 5 to check for potential improvement, done while adding low pt tight ID SF
    # Raise back to pt 10 to make simple for now
    # We cannot use no-iso because of the trigger SFs not available

    df = df.Define(
        "Muon_sel",
        """
        v_ops::pt(Muon_p4) > 10 && abs(v_ops::eta(Muon_p4)) < 2.4 && abs(Muon_dz) < 0.1 && abs(Muon_dxy) < 0.05 && abs(Muon_dxy) < 0.05 && Muon_sip3d <= 8 && Muon_pfIsoId >= 1 && Muon_looseId""",
    )
    # Can lower pT to 5 later when applying the soft muon SFs

    df = df.Define("Electron_iso", "Electron_mvaIso")
    df = df.Define("Muon_iso", "Muon_pfRelIso04_all")

    cand_columns = []
    for ch in channels:
        legs = getChannelLegs(ch)
        cand_column = f"HwwCandidates_{ch}"
        leg_args = []
        for leg_idx, leg in enumerate(legs):
            leg_args.extend(
                [
                    f"{leg}_sel",
                    f"{leg}_p4",
                    f"{leg}_iso",
                    f"{leg}_charge",
                    f"{leg}_genMatchIdx",
                ]
            )
        leg_str = ", ".join(leg_args)
        df = df.Define(cand_column, f"GetHWWCandidates(Channel::{ch}, 0.1, {leg_str})")
        cand_columns.append(cand_column)
    cand_filters = [f"{c}.size() > 0" for c in cand_columns]
    stringfilter = " || ".join(cand_filters)
    df = df.Filter(stringfilter, "Reco Baseline 2")
    cand_list_str = ", ".join(["&" + c for c in cand_columns])
    df = df.Define("HwwCandidate", f"GetBestHWWCandidate({{ {cand_list_str} }}, event)")
    df = df.Define("is_SL", "HwwCandidate.leg_type.size() == 1")
    return df


def RecoHWWJetSelection(df):
    df = df.Define(
        "Jet_Incl",
        f"v_ops::pt(Jet_p4)>20 && abs(v_ops::eta(Jet_p4)) < 2.5 && ( Jet_jetId & 2 )",
    )
    df = df.Define(
        "FatJet_Incl",
        "(v_ops::pt(FatJet_p4)>200 && abs(v_ops::eta(FatJet_p4)) < 2.5 ) && ( FatJet_jetId & 2 ) && (FatJet_msoftdrop > 30) ",
    )
    df = df.Define(
        "Jet_sel",
        """return RemoveOverlaps(Jet_p4, Jet_Incl,HwwCandidate.getLegP4s(), 0.4);""",
    )
    df = df.Define(
        "FatJet_sel",
        """return RemoveOverlaps(FatJet_p4, FatJet_Incl,HwwCandidate.getLegP4s(), 0.8);""",
    )
    df = df.Define(
        "FatJet_cleaned",
        " RemoveOverlaps(FatJet_p4, FatJet_sel, Jet_p4[Jet_sel], 0.8)",
    )

    df = df.Define(
        "n_eff_Jets", "(FatJet_p4[FatJet_cleaned].size()*2)+(Jet_p4[Jet_sel].size())"
    )

    # Lower preselected nJets to be >= 1 to prepare for fake estimation regions
    return df.Filter("n_eff_Jets>=1", "Reco jet candidates")
