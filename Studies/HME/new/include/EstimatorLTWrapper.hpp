#ifndef ESTIM_LT_WRAP
#define ESTIM_LT_WRAP

#include <memory>
#include "TString.h"
#include "Estimator.hpp"

namespace HME
{
    inline static const TString pdf_sl_name = "pdf_sl.root";
    inline static const TString pdf_dl_name = "pdf_dl.root";

    class EstimatorLTWrapper
    {
        private:
        EstimatorLTWrapper() {}
        std::unique_ptr<Estimator> estimator = std::make_unique<Estimator>(pdf_sl_name, pdf_dl_name);
        
        public:
        ~EstimatorLTWrapper() = default;
        static EstimatorLTWrapper& Instance()
        {
            static EstimatorLTWrapper ew;
            return ew;
        }

        Estimator& GetEstimator()
        {
            auto& pest = Instance().estimator;
            return *pest;
        }

        EstimatorLTWrapper(EstimatorLTWrapper const& other) = delete;
        EstimatorLTWrapper(EstimatorLTWrapper&& other) = delete;
        EstimatorLTWrapper& operator=(EstimatorLTWrapper const& other) = delete;
        EstimatorLTWrapper& operator=(EstimatorLTWrapper&& other) = delete;
    };
}

#endif