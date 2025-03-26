#ifndef ESTIM_SL
#define ESTIM_SL

#include "EstimatorBase.hpp"
#include "Constants.hpp"
#include "Definitions.hpp"
// #include "EstimatorUtils.hpp"

namespace HME 
{
    class EstimatorSingleLep final : public EstimatorBase
    {
        public:
        explicit EstimatorSingleLep(TString const& pdf_file_name);
        ~EstimatorSingleLep() override = default;

        ArrF_t<ESTIM_OUT_SZ> EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label) override;
        OptArrF_t<ESTIM_OUT_SZ> EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id) override;
    };

    ArrF_t<ESTIM_OUT_SZ> EstimatorSingleLep::EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label)
    {
        ArrF_t<ESTIM_OUT_SZ> res{};
        return res;
    }

    OptArrF_t<ESTIM_OUT_SZ> EstimatorSingleLep::EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id)
    {
        return std::nullopt;
    }

    EstimatorSingleLep::EstimatorSingleLep(TString const& pdf_file_name)
    {}
}

#endif