#ifndef EST_RCDR_HPP
#define EST_RCDR_HPP

#include <memory>

#include "TFile.h"
#include "TTree.h"

namespace HME
{
    class EstimationRecorder
    {
        public:
        explicit EstimationRecorder(TString const& dbg_file_name);

        inline void ResetTree(std::unique_ptr<TTree> tree) { m_tree = std::move(tree); }
        inline void FillTree() { m_tree->Fill(); }
        inline void WriteTree(TString const& tree_name) { m_file->WriteObject(m_tree.get(), tree_name); }
        inline bool ShouldRecord() { return m_file != nullptr; }

        private:
        std::unique_ptr<TFile> m_file;
        std::unique_ptr<TTree> m_tree;
    };

    EstimationRecorder::EstimationRecorder(TString const& dbg_file_name)
    {
        if (!dbg_file_name.EqualTo(""))
        {
            m_file = std::make_unique<TFile>(dbg_file_name, "RECREATE");
        }
    }
}

#endif