#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cstddef>
#include <unordered_map>

#include "TString.h"

namespace HME 
{
    inline constexpr int SEED = 42;

    enum class Channel { SL, DL };
    enum class Lep { lep1, lep2, count };
    enum class Nu { nu1, nu2, count };
    enum class Quark { b1, b2, q1, q2, count };
    inline constexpr size_t MAX_GEN_LEP = static_cast<size_t>(Lep::count);
    inline constexpr size_t MAX_GEN_QUARK = static_cast<size_t>(Quark::count);
    inline constexpr size_t MAX_GEN_NU = static_cast<size_t>(Nu::count);

    // PDFs in SL channel resolved topology
    enum class PDF1_sl { b1, q1, numet_pt, numet_dphi, nulep_deta, hh_dphi, mbb, mww, hh_deta, count };
    enum class PDF2_sl { b1b2, q1q2, mw1mw2, hh_dEtadPhi, hh_pt_e, count };
    inline constexpr size_t NUM_PDF_1D_SL = static_cast<size_t>(PDF1_sl::count);
    inline constexpr size_t NUM_PDF_2D_SL = static_cast<size_t>(PDF2_sl::count);

    // PDFs in DL channel resolved topology
    enum class PDF1_dl { b1, mw_onshell, count };
    enum class PDF2_dl { count };
    inline constexpr size_t NUM_PDF_1D_DL = static_cast<size_t>(PDF1_dl::count);
    inline constexpr size_t NUM_PDF_2D_DL = static_cast<size_t>(PDF2_dl::count);

    // return values of Estimator
    enum class EstimOut { mass, integral, width, peak_value, count };
    inline constexpr size_t ESTIM_OUT_SZ = static_cast<size_t>(EstimOut::count);

    // objects
    enum class ObjSL { bj1, bj2, lj1, lj2, lep, met, count };
    enum class ObjDL { bj1, bj2, lep1, lep2, met, count };

    inline constexpr size_t MAX_GEN_JET = 20;
    inline constexpr size_t MAX_RECO_JET = 20;
    inline constexpr size_t MAX_RECO_LEP = 2;
    inline constexpr size_t NUM_BQ = 2;
    inline constexpr size_t NUM_LQ = 2;

    inline constexpr Float_t HIGGS_MASS = 125.03;
    inline constexpr Float_t HIGGS_WIDTH = 0.004;
    inline constexpr Float_t TOL = 10e-7;
    inline constexpr int N_ATTEMPTS = 1;
    inline constexpr int N_ITER = 1000;
    inline constexpr int CONTROL = 4;

    inline constexpr Float_t MAX_MASS = 4000.0;
    inline constexpr Float_t MIN_MASS = 200.0;
    inline constexpr int N_BINS = 3800;

    inline constexpr Float_t MET_SIGMA = 25.2;
    inline constexpr Float_t DEFAULT_JET_RES = 0.1;

    inline constexpr unsigned Q16 = 16;
    inline constexpr unsigned Q84 = 84;

    inline constexpr size_t NUM_BEST_BTAG = 2;

    inline static const std::unordered_map<PDF1_sl, TString> pdf1d_sl_names = { { PDF1_sl::numet_pt, "pdf_numet_pt" },
                                                                                { PDF1_sl::numet_dphi, "pdf_numet_dphi" },
                                                                                { PDF1_sl::nulep_deta, "pdf_nulep_deta" },
                                                                                { PDF1_sl::hh_dphi, "pdf_hh_dphi" },
                                                                                { PDF1_sl::mbb, "pdf_mbb" },
                                                                                { PDF1_sl::mww, "pdf_mww_narrow" },
                                                                                { PDF1_sl::hh_deta, "pdf_hh_deta" },
                                                                                { PDF1_sl::q1, "pdf_q1" },
                                                                                { PDF1_sl::b1, "pdf_b1" } };

    inline static const std::unordered_map<PDF2_sl, TString> pdf2d_sl_names = { { PDF2_sl::b1b2, "pdf_b1b2" },
                                                                                { PDF2_sl::q1q2, "pdf_q1q2" },
                                                                                { PDF2_sl::mw1mw2, "pdf_mw1mw2" },
                                                                                { PDF2_sl::hh_dEtadPhi, "pdf_hh_dEtadPhi" },
                                                                                { PDF2_sl::hh_pt_e, "pdf_hh_pt_e" } };

    inline static const std::unordered_map<PDF1_dl, TString> pdf1d_dl_names = { { PDF1_dl::b1, "pdf_b1_run2" }, 
                                                                                { PDF1_dl::mw_onshell, "pdf_mw_onshell" }};
                                                                            
    inline static const std::unordered_map<PDF2_dl, TString> pdf2d_dl_names = {};

}

#endif