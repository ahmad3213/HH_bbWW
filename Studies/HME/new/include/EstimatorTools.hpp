#ifndef ESTIMATOR_TOOLS_HPP
#define ESTIMATOR_TOOLS_HPP

#include "TVector2.h"
#include "TRandom3.h"
#include "Definitions.hpp"

namespace HME 
{
    inline Float_t mT(LorentzVectorF_t p)
    {
        return std::sqrt(p.M2() + p.Pt()*p.Pt());
    }

    LorentzVectorF_t SamplePNetResCorr(LorentzVectorF_t const& jet, std::unique_ptr<TRandom3>& prg, Float_t resolution)
    {
        Float_t dpt = prg->Gaus(0.0, resolution);
        Float_t pt = jet.Pt();
        while (pt + dpt < 0.0)
        {
            dpt = prg->Gaus(0.0, resolution);
        }
        return LorentzVectorF_t(pt + dpt, jet.Eta(), jet.Phi(), jet.M());
    }

    OptPairF_t ComputeJetResc(LorentzVectorF_t const& p1, LorentzVectorF_t const& p2, UHist_t<TH1F>& pdf, Float_t mass)
    {
        Float_t c1 = pdf->GetRandom();
        Float_t x1 = p2.M2();
        Float_t x2 = 2.0*c1*(p1.Dot(p2));
        Float_t x3 = c1*c1*p1.M2() - mass*mass;
        Float_t discrim = x2*x2 - 4.0*x1*x3;
        if (x2 >= 0.0 && x1 != 0.0 && discrim >= 0.0)
        {
            Float_t c2 = (-x2 + std::sqrt(discrim))/(2.0*x1);
            if (c2 >= 0.0)
            {
                return std::make_optional<std::pair<Float_t, Float_t>>(c1, c2); 
            }
        }
        return std::nullopt;
    }

    OptLVecF_t NuFromOnshellW(Float_t eta, Float_t phi, Float_t mw, LorentzVectorF_t const& lep_onshell)
    {
        Float_t deta = eta - lep_onshell.Eta();
        Float_t dphi = phi - lep_onshell.Phi();
        Float_t pt = mw*mw/(2.0*lep_onshell.Pt()*(std::cosh(deta) - std::cos(dphi)));

        if (std::isinf(pt) || std::isnan(pt))
        {
            return std::nullopt;
        }
        return std::make_optional<LorentzVectorF_t>(pt, eta, phi, 0.0);
    }

    OptLVecF_t NuFromOffshellW(LorentzVectorF_t const& lep1, 
                            LorentzVectorF_t const& lep2, 
                            LorentzVectorF_t const& nu1,
                            LorentzVectorF_t const& met,
                            int control, 
                            Float_t mh)
    {
        LorentzVectorF_t tmp = lep1 + lep2 + nu1;

        // how much is contribution of nu2 in met
        // assumption: met = nu1 + nu2
        Float_t nu_tmp_px = met.Px() - nu1.Px();
        Float_t nu_tmp_py = met.Py() - nu1.Py();

        // weird stuff 
        Float_t px = std::sqrt(tmp.Pt()*tmp.Pt() + tmp.M2());
        Float_t py = 0.0;
        Float_t pz = tmp.Pz();
        Float_t en = tmp.E();

        LorentzVectorF_t tmp2;
        tmp2.SetPxPyPzE(px, py, pz, en);

        // pt and phi of nu2 are determined by difference of met and transverse component of nu1
        Float_t pt = std::sqrt(nu_tmp_px*nu_tmp_px + nu_tmp_py*nu_tmp_py);
        Float_t phi = std::atan2(nu_tmp_py, nu_tmp_px);

        Float_t cosh_deta = (mh*mh + 2.0*(nu_tmp_px*tmp.Px() + nu_tmp_py*tmp.Py()) - tmp.M2())/(2.0*tmp2.Pt()*pt);
        if (cosh_deta < 1.0)
        {
            return std::nullopt;
        }

        Float_t delta_eta = std::acosh(cosh_deta);
        Float_t eta = control == 1 ? tmp2.Eta() - delta_eta : tmp2.Eta() + delta_eta;

        if (std::abs(eta) > 7.0)
        {
            return std::nullopt;
        }

        return std::make_optional<LorentzVectorF_t>(pt, eta, phi, 0.0);
    }

    OptLVecF_t NuFromH(LorentzVectorF_t const& jet1, 
                    LorentzVectorF_t const& jet2, 
                    LorentzVectorF_t const& lep, 
                    LorentzVectorF_t const& met, 
                    bool add_deta,
                    Float_t mh)
    {
        LorentzVectorF_t vis = jet1 + jet2 + lep;
        Float_t cos_dphi = std::cos(TVector2::Phi_mpi_pi(vis.Phi() - met.Phi())); 
        Float_t mt = mT(vis);
        Float_t mv2 = vis.M2();
        Float_t pt = met.Pt(); 
        Float_t phi = met.Phi();   
        // Float_t cosh_deta = (mh*mh - mv2)/(2.0*pt*mt) + (vis.Pt()/mt)*cos_dphi;
        // if (cosh_deta < 1.0)
        // {
        //     return std::nullopt;
        // }
        // Float_t delta_eta = std::acosh(cosh_deta);
        // Float_t eta = add_deta ? vis.Eta() + delta_eta : vis.Eta() - delta_eta;
        // if (std::abs(eta) > 7.0)
        // {
        //     return std::nullopt;
        // }
        // return std::make_optional<LorentzVectorF_t>(pt, eta, phi, 0.0);
        Float_t cosh_dy = (mh*mh - mv2)/(2.0*pt*mt) + (vis.Pt()/mt)*cos_dphi;
        if (cosh_dy < 1.0)
        {
            return std::nullopt;
        }
        Float_t dy = std::acosh(cosh_dy);
        Float_t y = add_deta ? vis.ColinearRapidity() + dy : vis.ColinearRapidity() - dy;
        if (std::abs(y) > 7.0)
        {
            return std::nullopt;
        }
        return std::make_optional<LorentzVectorF_t>(pt, y, phi, 0.0);
    }

    // std::optional<LorentzVectorF_t> NuFromHiggsConstr(LorentzVectorF_t const& jet1, 
    //                                                   LorentzVectorF_t const& jet2, 
    //                                                   LorentzVectorF_t const& lep, 
    //                                                   LorentzVectorF_t const& met, 
    //                                                   int control, 
    //                                                   Float_t mh)
    // {
    //     LorentzVectorF_t tmp = jet1 + jet2 + lep;
    //     Float_t nu_tmp_px = met.Px();
    //     Float_t nu_tmp_py = met.Py();

    //     // weird stuff 
    //     Float_t px = std::sqrt(tmp.Pt()*tmp.Pt() + tmp.M2());
    //     Float_t py = 0.0;
    //     Float_t pz = tmp.Pz();
    //     Float_t en = tmp.E();

    //     LorentzVectorF_t tmp2;
    //     tmp2.SetPxPyPzE(px, py, pz, en);

    //     Float_t pt = std::sqrt(nu_tmp_px*nu_tmp_px + nu_tmp_py*nu_tmp_py);
    //     Float_t phi = std::atan2(nu_tmp_py, nu_tmp_px);

    //     Float_t cosh_deta = (mh*mh + 2.0*(nu_tmp_px*tmp.Px() + nu_tmp_py*tmp.Py()) - tmp.M2())/(2.0*tmp2.Pt()*pt);
    //     if (cosh_deta < 1.0)
    //     {
    //         return std::nullopt;
    //     }

    //     Float_t delta_eta = std::acosh(cosh_deta);
    //     Float_t eta = control == 1 ? tmp2.Eta() - delta_eta : tmp2.Eta() + delta_eta;

    //     if (std::abs(eta) > 7.0)
    //     {
    //         return std::nullopt;
    //     }

    //     return std::make_optional<LorentzVectorF_t>(pt, eta, phi, 0.0);
    // }

    OptLVecF_t NuFromW(LorentzVectorF_t const& lep, LorentzVectorF_t const& met, bool add_deta, Float_t mw)
    {
        Float_t pt = met.Pt();
        Float_t phi = met.Phi();
        Float_t dphi = TVector2::Phi_mpi_pi(phi - lep.Phi());
        Float_t cosh_deta = mw*mw/(2.0*lep.Pt()*pt) + std::cos(dphi);    
        if (cosh_deta < 1.0)
        {
            return std::nullopt;
        }
        Float_t delta_eta = std::acosh(cosh_deta);
        Float_t eta = add_deta ? lep.Eta() + delta_eta : lep.Eta() - delta_eta;
        if (std::abs(eta) > 7.0)
        {
            return std::nullopt;
        }
        return std::make_optional<LorentzVectorF_t>(pt, eta, phi, 0.0);
    }

    OptLVecFPair_t NuFromConstraints(LorentzVectorF_t const& jet1, LorentzVectorF_t const& jet2, 
                                    LorentzVectorF_t const& lep, LorentzVectorF_t const& met, 
                                    Float_t mw, Float_t mh)
    {
        LorentzVectorF_t vis = jet1 + jet2 + lep;
        Float_t phi = met.Phi();
        Float_t cos_dphi_nulep = std::cos(TVector2::Phi_mpi_pi(phi - lep.Phi()));
        Float_t cos_dphi_nuvis = std::cos(TVector2::Phi_mpi_pi(phi - vis.Phi()));

        Float_t A = (mh*mh - vis.M2())*lep.Pt();
        Float_t B = mw*mw*mT(vis);
        Float_t C = (mh*mh - vis.M2())*lep.Pt()*cos_dphi_nulep - mw*mw*vis.Pt()*cos_dphi_nuvis;

        Float_t D = A*std::cosh(lep.ColinearRapidity()) - B*std::cosh(vis.ColinearRapidity());
        Float_t E = -A*std::sinh(lep.ColinearRapidity()) + B*std::sinh(vis.ColinearRapidity());

        // eqn trying to solve:
        // (D - E)x^2 - 2Cx + (D + E) = 0
        // x = exp(-y), y - RAPIDITY, not pseudorapidity 
        // x > 0.0
        // however, for masseless particles rapididyt = pseudorapidity
        // but in all other calculation the above is not true

        // linear equation
        if (D - E == 0.0)
        {
            if (C == 0.0)
            {
                return std::nullopt;
            }
            Float_t exp_neg_y = (D + E)/(2.0*C);
            if (exp_neg_y < 0.0)
            {
                return std::nullopt;
            }
            Float_t y = -std::log(exp_neg_y);
            Float_t cosh_dy_nulep = std::cosh(y - lep.ColinearRapidity());
            Float_t pt = mw*mw/(2.0*lep.Pt()*(cosh_dy_nulep - cos_dphi_nulep));
            if (pt < 0.0 || std::isnan(pt) || std::isinf(pt))
            {
                return std::nullopt;
            }
            LorentzVectorF_t nu = LorentzVectorF_t(pt, y, phi, 0.0);
            return std::make_optional<std::pair<LorentzVectorF_t, LorentzVectorF_t>>(nu, nu);
        }

        // quardratic: no solutions
        Float_t disc2 = C*C - D*D + E*E;
        if (disc2 < 0.0)
        {
            return std::nullopt;
        }

        // quadratic: two equal solutions
        if (disc2 == 0.0)
        {
            Float_t exp_neg_y = C/(D - E);
            if (exp_neg_y < 0.0)
            {
                return std::nullopt;
            }
            Float_t y = -std::log(exp_neg_y);
            Float_t cosh_dy_nulep = std::cosh(y - lep.ColinearRapidity());
            Float_t pt = mw*mw/(2.0*lep.Pt()*(cosh_dy_nulep - cos_dphi_nulep));
            if (pt < 0.0 || std::isnan(pt) || std::isinf(pt))
            {
                return std::nullopt;
            }
            LorentzVectorF_t nu = LorentzVectorF_t(pt, y, phi, 0.0);
            return std::make_optional<std::pair<LorentzVectorF_t, LorentzVectorF_t>>(nu, nu);
        }

        // quadratic: two different solutions
        Float_t exp_neg_y1 = (C + std::sqrt(disc2))/(D - E);
        Float_t exp_neg_y2 = (C - std::sqrt(disc2))/(D - E);
        if (exp_neg_y1 < 0.0 && exp_neg_y2 < 0.0)
        {
            return std::nullopt;
        }

        LorentzVectorF_t nu1, nu2;
        if (exp_neg_y1 > 0.0)
        {
            Float_t y1 = -std::log(exp_neg_y1);
            Float_t cosh_dy_nulep = std::cosh(y1 - lep.ColinearRapidity());
            Float_t pt = mw*mw/(2.0*lep.Pt()*(cosh_dy_nulep - cos_dphi_nulep));
            bool invalid_pt = pt < 0.0 || std::isnan(pt) || std::isinf(pt);
            if (!invalid_pt)
            {
                nu1 = LorentzVectorF_t(pt, y1, phi, 0.0);
            }
        }
        
        if (exp_neg_y2 > 0.0)
        {
            Float_t y2 = -std::log(exp_neg_y2);
            Float_t cosh_dy_nulep = std::cosh(y2 - lep.ColinearRapidity());
            Float_t pt = mw*mw/(2.0*lep.Pt()*(cosh_dy_nulep - cos_dphi_nulep));
            bool invalid_pt = pt < 0.0 || std::isnan(pt) || std::isinf(pt);
            if (!invalid_pt)
            {
                nu2 = LorentzVectorF_t(pt, y2, phi, 0.0);
            }
        }

        return std::make_optional<std::pair<LorentzVectorF_t, LorentzVectorF_t>>(nu1, nu2);
    }
}

#endif