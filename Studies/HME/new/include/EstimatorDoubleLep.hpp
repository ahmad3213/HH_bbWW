#ifndef ESTIM_DL
#define ESTIM_DL

#include "EstimatorBase.hpp"
#include "Constants.hpp"
#include "Definitions.hpp"
#include "EstimatorUtils.hpp"
#include "EstimatorTools.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "TVector2.h"
#include "Math/GenVector/VectorUtil.h" // DeltaPhi
using ROOT::Math::VectorUtil::DeltaPhi;

namespace HME 
{
    class EstimatorDoubleLep final : public EstimatorBase
    {
        public:
        explicit EstimatorDoubleLep(TString const& pdf_file_name);
        ~EstimatorDoubleLep() override = default;

        ArrF_t<ESTIM_OUT_SZ> EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label) override;
        OptArrF_t<ESTIM_OUT_SZ> EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id) override;
    };

    EstimatorDoubleLep::EstimatorDoubleLep(TString const& pdf_file_name)
    {
        m_pdf_1d.resize(pdf1d_dl_names.size());
        m_pdf_2d.resize(pdf2d_dl_names.size());
        
        TFile* pf = TFile::Open(pdf_file_name);
        Get1dPDFs(pf, m_pdf_1d, Channel::DL);
        Get2dPDFs(pf, m_pdf_2d, Channel::DL);
        pf->Close();
    }

    ArrF_t<ESTIM_OUT_SZ> EstimatorDoubleLep::EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label)
    {
        ArrF_t<ESTIM_OUT_SZ> res{};
        std::fill(res.begin(), res.end(), -1.0f);

        LorentzVectorF_t const& bj1 = particles[static_cast<size_t>(ObjDL::bj1)];
        LorentzVectorF_t const& bj2 = particles[static_cast<size_t>(ObjDL::bj2)];
        LorentzVectorF_t const& lep1 = particles[static_cast<size_t>(ObjDL::lep1)];
        LorentzVectorF_t const& lep2 = particles[static_cast<size_t>(ObjDL::lep2)];
        LorentzVectorF_t const& met = particles[static_cast<size_t>(ObjDL::met)];

        UHist_t<TH1F>& pdf_b1 = m_pdf_1d[static_cast<size_t>(PDF1_dl::b1)];
        UHist_t<TH1F>& pdf_mw_onshell = m_pdf_1d[static_cast<size_t>(PDF1_dl::mw_onshell)];

        m_res_mass->SetNameTitle("X_mass", Form("X->HH mass: event %llu, comb %s", evt_id, comb_label.Data()));

        for (int i = 0; i < N_ITER; ++i)
        {
            Float_t eta_gen = m_prg->Uniform(-6, 6);
            Float_t phi_gen = m_prg->Uniform(-3.1415926, 3.1415926);
            Float_t mh = m_prg->Gaus(HIGGS_MASS, HIGGS_WIDTH);
            Float_t mw = pdf_mw_onshell->GetRandom(m_prg.get());
            Float_t smear_dpx = m_prg->Gaus(0.0, MET_SIGMA);
            Float_t smear_dpy = m_prg->Gaus(0.0, MET_SIGMA);

            auto bresc = ComputeJetResc(bj1, bj2, pdf_b1, mh);
            if (!bresc.has_value())
            {
                continue;
            }
            auto [c1, c2] = bresc.value();

            LorentzVectorF_t b1 = bj1;
            LorentzVectorF_t b2 = bj2;
            b1 *= c1;
            b2 *= c2;

            Float_t jet_resc_dpx = -1.0*(c1 - 1)*bj1.Px() - (c2 - 1)*bj2.Px();
            Float_t jet_resc_dpy = -1.0*(c1 - 1)*bj1.Py() - (c2 - 1)*bj2.Py();

            Float_t met_corr_px = met.Px() + jet_resc_dpx + smear_dpx;
            Float_t met_corr_py = met.Py() + jet_resc_dpy + smear_dpy;

            Float_t met_corr_pt = std::sqrt(met_corr_px*met_corr_px + met_corr_py*met_corr_py);
            Float_t met_corr_phi = std::atan2(met_corr_py, met_corr_px);
            LorentzVectorF_t met_corr(met_corr_pt, 0.0, met_corr_phi, 0.0);

            LorentzVectorF_t Hbb = b1;
            Hbb += b2;

            std::vector<Float_t> estimates;
            // two options: 
            // lep1 comes from onshell W and lep2 comes from offshell W
            // lep1 comes from offshell W and lep2 comes from onshell W
            // when neutrino is computed in each case again two options are possible: delta_eta is added or subtracted to nu
            // in total: 4 combinations; they are encoded in this for loop
            for (int j = 0; j < CONTROL; ++j)
            {
                LorentzVectorF_t l_offshell;
                LorentzVectorF_t l_onshell;
                
                int is_onshell = j / 2;
                if (is_onshell == 0) 
                {
                    l_onshell = lep1;
                    l_offshell = lep2;
                } 
                else 
                {
                    l_onshell = lep2;
                    l_offshell = lep1;
                }

                auto nu_onshell = NuFromOnshellW(eta_gen, phi_gen, mw, l_onshell);  
                if (!nu_onshell)
                {
                    continue;
                }

                int is_offshell = j % 2;
                auto nu_offshell = NuFromOffshellW(lep1, lep2, nu_onshell.value(), met_corr, is_offshell, mh);
                if (!nu_offshell)
                {
                    continue;
                }

                LorentzVectorF_t onshellW = l_onshell + nu_onshell.value();
                LorentzVectorF_t offshellW = l_offshell + nu_offshell.value();
                LorentzVectorF_t Hww = onshellW + offshellW;

                if (offshellW.M() > mh/2)
                {
                    continue;
                }

                if (std::abs(Hww.M() - mh) > 1.0)
                {
                    continue;
                }

                Float_t mX = (Hbb + Hww).M();
                if (mX > 0.0)
                {
                    estimates.push_back(mX);
                }
            }

            Float_t weight = estimates.empty() ? 0.0 : 1.0/estimates.size();
            for (auto est: estimates)
            {
                m_res_mass->Fill(est, weight);
            }
        }

        Float_t integral = m_res_mass->Integral();
        if (m_res_mass->GetEntries() && integral > 0.0)
        {
            int binmax = m_res_mass->GetMaximumBin(); 
            res[static_cast<size_t>(EstimOut::mass)] = m_res_mass->GetXaxis()->GetBinCenter(binmax);
            res[static_cast<size_t>(EstimOut::peak_value)] = m_res_mass->GetBinContent(binmax);
            res[static_cast<size_t>(EstimOut::width)] = ComputeWidth(m_res_mass, Q16, Q84);
            res[static_cast<size_t>(EstimOut::integral)] = integral;
            return res;
        }
        return res;
    }

    OptArrF_t<ESTIM_OUT_SZ> EstimatorDoubleLep::EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id)
    {
        VecLVF_t particles(static_cast<size_t>(ObjDL::count));
        particles[static_cast<size_t>(ObjDL::lep1)] = leptons[static_cast<size_t>(Lep::lep1)];
        particles[static_cast<size_t>(ObjDL::lep2)] = leptons[static_cast<size_t>(Lep::lep2)];
        particles[static_cast<size_t>(ObjDL::met)] = met;

        std::vector<ArrF_t<ESTIM_OUT_SZ>> estimations;
        size_t num_bjets = jets.size() < NUM_BEST_BTAG ? jets.size() : NUM_BEST_BTAG;
        
        for (size_t bj1_idx = 0; bj1_idx < num_bjets; ++bj1_idx)
        {
            for (size_t bj2_idx = bj1_idx + 1; bj2_idx < num_bjets; ++bj2_idx)
            {
                // order jets such that first b jet has bigger pt and save their p4
                if (jets[bj1_idx].Pt() > jets[bj2_idx].Pt())
                {
                    particles[static_cast<size_t>(ObjDL::bj1)] = jets[bj1_idx];
                    particles[static_cast<size_t>(ObjDL::bj2)] = jets[bj2_idx];
                }
                else 
                {
                    particles[static_cast<size_t>(ObjDL::bj1)] = jets[bj2_idx];
                    particles[static_cast<size_t>(ObjDL::bj2)] = jets[bj1_idx];
                }

                TString comb_label = Form("b%zub%zu", bj1_idx, bj2_idx);
                ArrF_t<ESTIM_OUT_SZ> comb_result = EstimateCombination(particles, evt_id, comb_label);

                // success: mass > 0
                if (comb_result[static_cast<size_t>(EstimOut::mass)] > 0.0)
                {
                    estimations.push_back(comb_result);
                }

                // clear the histogram to be reused 
                ResetHist(m_res_mass);
            }
        }

        // success: at least one combination produced an estimate of X->HH mass
        if (!estimations.empty())
        {
            return std::make_optional<ArrF_t<ESTIM_OUT_SZ>>(estimations[0]);
        }
        return std::nullopt;
    }
}

#endif