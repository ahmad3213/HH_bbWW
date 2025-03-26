#ifndef ESTIM_BASE
#define ESTIM_BASE

#include <optional>
#include <type_traits>

#include "TRandom3.h"

#include "Definitions.hpp"
#include "EstimationRecorder.hpp"
#include "Constants.hpp"

namespace HME 
{
    template <typename T, std::enable_if_t<std::is_default_constructible_v<T>, bool> = true>
    void ResetObject(T& object)
    {
        object = T{};
    }

    class EstimatorBase
    {
        public:
        EstimatorBase();
        virtual ~EstimatorBase() = default;

        virtual ArrF_t<ESTIM_OUT_SZ> EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label) = 0;
        virtual OptArrF_t<ESTIM_OUT_SZ> EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id) = 0;

        protected:
        HistVec_t<TH1F> m_pdf_1d;
        HistVec_t<TH2F> m_pdf_2d;
        std::unique_ptr<TRandom3> m_prg;
        UHist_t<TH1F> m_res_mass;
    };

    EstimatorBase::EstimatorBase() 
    :   m_prg(std::make_unique<TRandom3>(SEED))
    ,   m_res_mass(std::make_unique<TH1F>("none", "none", N_BINS, MIN_MASS, MAX_MASS))
    {}
}

#endif