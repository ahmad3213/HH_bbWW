#ifndef DEF_HPP
#define DEF_HPP

#include <vector>
#include <memory>
#include <optional>

#include "Math/GenVector/LorentzVector.h"
#include "Math/Vector4D.h"

#include "TH1.h"
#include "TH2.h"

#include "Constants.hpp"

namespace HME
{
    template <typename T> 
    using UHist_t = std::unique_ptr<T>;

    template <typename T> 
    using HistVec_t = std::vector<UHist_t<T>>;

    using LorentzVectorF_t = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Float_t>>;
    using VecLVF_t = std::vector<LorentzVectorF_t>;

    template <size_t N>
    using ArrF_t = std::array<Float_t, N>;

    template <size_t N>
    using OptArrF_t = std::optional<ArrF_t<N>>;

    using OptPairF_t = std::optional<std::pair<Float_t, Float_t>>;
    using OptLVecF_t = std::optional<LorentzVectorF_t>;
    using OptLVecFPair_t = std::optional<std::pair<LorentzVectorF_t, LorentzVectorF_t>>;
}

#endif