#pragma once

#include <map>
#include <vector>
#include <thread>
#include <typeinfo>
#include <variant>

#include "EntryQueue.h"


using RVecF = ROOT::VecOps::RVec<float>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecUC = ROOT::VecOps::RVec<unsigned char>;
using RVecShort = ROOT::VecOps::RVec<short>;




namespace analysis {
typedef std::variant<int,float,bool, unsigned long,unsigned long long,long,unsigned int, unsigned char, RVecI, RVecF, RVecUC, RVecShort> MultiType;

struct Entry{
    std::vector<MultiType> var_values;

    explicit Entry(size_t size) : var_values(size) {}

    template<typename T>
    void Add(int index, const T& value){
        if (index >= 0){
            var_values.at(index) = value;
        }
    }

    template<typename T>
    const T& GetValue(int idx) const {
        return std::get<T>(var_values.at(idx));
    }
};

namespace detail {
    inline void putEntry(std::shared_ptr<Entry>& entry, int index) {}

    template<typename T, typename ...Args>
    //void putEntry(std::vector<Entry>& entries, int var_index, const T& value, Args&& ...args){
    void putEntry(std::shared_ptr<Entry>& entry, int var_index, const T& value, Args&& ...args){

        // std::cout << "Var index is " << var_index << std::endl;
        // std::cout << "And value is " << value << std::endl;
        entry->Add(var_index, value);
        putEntry(entry, var_index+1, std::forward<Args>(args)...);
    }

    inline void read(int index) {}

    template<typename T, typename ...Args>
    void read(int var_index, const T& value, Args&& ...args){
        //index 138 is lep1_pt, lets only look at that one
        if (var_index == 138){
            std::cout << "Var index is " << var_index << std::endl;
            std::cout << "And value is " << value << std::endl;
        }
        read(var_index+1, std::forward<Args>(args)...);
    }


}

struct StopLoop {};

template<typename ...Args> //Using a template will allow us to pass the column types as a 'variable'! Trop Cool!
struct TupleMaker{
    TupleMaker(size_t queue_size, size_t max_entries)
        : queue(queue_size, max_entries) {
            std::cout << "Initializing tuplemaker with queue size " << queue_size << " and max entries " << max_entries << std::endl;
        }


    void readDF(ROOT::RDF::RNode df, const std::vector<std::string>& column_names){
        int entry_counter = 0;
        df.Foreach([&](const Args& ...args){
            std::cout << "New entry! " << entry_counter << std::endl;
            entry_counter++;
            detail::read(0, args...);
            std::cout << "End of entry" << std::endl;
        }, column_names);
    }

    
    ROOT::RDF::RNode FillDF(ROOT::RDF::RNode new_df, ROOT::RDF::RNode in_df, const std::vector<int>& local_to_master_map, const int master_size, const std::vector<std::string>& local_column_names, int nBatchStart, int nBatchEnd, int batch_size){


        auto df0 = in_df.Define("_entry", [=](const Args& ...args) {
            // auto entry = std::make_shared<Entry>(local_column_names.size());
            auto entry = std::make_shared<Entry>(master_size);
            int index = 0;
            (void) std::initializer_list<int>{ (entry->Add(local_to_master_map.at(index++), args), 0)... };

            return entry;
        }, local_column_names);

        thread = std::make_unique<std::thread>([=]() {
            std::cout << "TupleMaker::FillDF: thread started." << std::endl;
            try{
                //std::cout << "Passed the lock step, starting foreach" << std::endl;
                ROOT::RDF::RNode df = df0;

                // df.Foreach([&](const Args& ...args){
                //     auto entry = std::make_shared<Entry>(column_names.size());
                //     detail::putEntry(entry, 0, args...);
                //     if(!queue.Push(entry)){
                //         std::cout << "Hey the push returned false" << std::endl;
                //         throw StopLoop();
                //     }
                // }, column_names);


                df.Foreach([&](const std::shared_ptr<Entry>& entry) {
                    if(!queue.Push(entry)){
                        std::cout << "Hey the push returned false" << std::endl;
                        throw StopLoop();
                    }
                }, {"_entry"});
            }
            catch(StopLoop){}

            std::cout << "Finished foreach" << std::endl;

            queue.SetInputAvailable(false);
        });
        

        new_df = new_df.Define("_entry", [=](ULong64_t rdfentry){
            // Entry entry(column_names.size());
            std::shared_ptr<Entry> entry;
            //int batch_size = 100;
            int start_idx = nBatchStart;
            int end_idx = nBatchEnd;
            const int index = rdfentry % batch_size;
            if (index >= start_idx && index < end_idx){
                queue.Pop(entry);
            }
            return entry;
        }, { "rdfentry_" });
        return new_df;


    }
    

    void join(){
        if (thread){
            queue.SetOutputNeeded(false);
            thread->join();
        }
    }



    EntryQueue<std::shared_ptr<Entry>> queue;
    std::unique_ptr<std::thread> thread;
    std::condition_variable cond_var;
};


} // namespace analysis