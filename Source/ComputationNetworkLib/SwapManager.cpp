//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapManager.h"
#include "SwapAction.h"
#include "../ComputationNetworkLib/ComputationNode.h"
#include "Sequences.h"
#include "SwapInAction.h"
#include "SwapOutAction.h"

#include <cmath>
#include <iostream>

bool g_useMemorySwapping = false;

namespace Microsoft { namespace MSR { namespace CNTK {


using std::cout;
using std::endl;

inline int SampleSize(){ return 100; }
inline int SwapSampleSize(){ return 10; }
inline float MeasurementUncertainty(){ return 1.15f; }


template <typename ElemType> SwapManager<ElemType>::SwapManager()
{
        m_useMemorySwapping = g_useMemorySwapping;
#ifndef CPUONLY
        size_t free, total;
        CUDA_CALL(cudaMemGetInfo(&free, &total));
        m_freeMemoryBase = Bytes2GB(free);
        m_totalMemoryBase = Bytes2GB(total);
#endif
}

template <typename ElemType> void SwapManager<ElemType>::CleanUp()
{
    // this releases the page-lock (or pinned) cudaHostmemory
    for(auto pair : m_buffer2SwapOut)
        pair.second->ReleaseMemory();
}


template<typename ElemType> void SwapManager<ElemType>::BeginSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining)
{

#ifndef CPUONLY
	if(!m_useMemorySwapping || !isTraining){ return; }

    // swap-in buffers which were not used in the backward pass
    if(!isForward)
        for(auto action : m_node2BackwardSwapin[node])
        {
            action->BeginAction();
            action->EndAction();
        }
#endif
}


template<typename ElemType> void SwapManager<ElemType>::EndSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining)
{
#ifndef CPUONLY
	if(!m_useMemorySwapping || !isTraining){ return; }

    // swap out in forward pass only; during the backward pass the memory is either (1) re-used for gradients, (2) freed
    if(isForward)
        for(auto action : m_node2ForwardSwapOut[node])
        {
            action->BeginAction();
            action->EndAction();
        }
    else
        for(auto matrix : m_node2BackwardFree[node])
        {
            // free memory during backward pass when no longer needed
            matrix->Resize(0,0,0,false);
        }
#endif
}

template <typename ElemType> float SwapManager<ElemType>::Bytes2GB(size_t bytes) 
{
    return ((float)bytes)/1024.0f/1024.0f/1024.0f;
}

template <typename ElemType> std::unordered_map<int,float> SwapManager<ElemType>::CumulativeMemory(int startStep, int endStep)
{
    int i = startStep;
    bool forwardMode = true;
    float swapInSize = 0.0f;
    float swapOutSize = 0.0f;
    float freedSize = 0.0f;
    std::unordered_map<int, float> step2CumulativeMemory;

    assert(endStep <= m_maxTimeStep);
    assert(startStep <= m_maxTimeStep);

    if(startStep == m_maxTimeStep){ forwardMode = false; }
    while(i != endStep)
    {

        ComputationNodeBase* node = m_timeStep2Node[i];

        if(forwardMode)
        {
            for(auto swapOut : m_node2ForwardSwapOut[node])
                swapOutSize += Bytes2GB(swapOut->GetGPUMatrix()->BufferSize());
        }
        else
        {
            for(auto swapIn : m_node2BackwardSwapin[node])
                swapInSize += Bytes2GB(swapIn->GetGPUMatrix()->BufferSize());

            for(auto freed : m_node2BackwardFree[node])
                freedSize += Bytes2GB(freed->BufferSize());
        }

        step2CumulativeMemory[i] = m_freeMemoryBase + swapInSize - swapOutSize - freedSize;

        if(m_timeStep2Node.count(i+1) == 0){ forwardMode = false; }
        if(forwardMode){ i++; }
        else{ i--; }

    }

    return step2CumulativeMemory;
}


template <typename ElemType> void SwapManager<ElemType>::InitializeSwapping(
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > forwardSwapOutNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > backwardSwapInNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > lastBackwardNodes2matrices, std::unordered_map<ComputationNodeBase*, int> node2TimeStep)
{

    ClearActionsAndTheirMemory();
    // setup swapout actions
    for(auto pair : forwardSwapOutNodes2matrices)
    {
        for(auto buffer : pair.second)
        {
            if(m_buffer2SwapOut.count(buffer) == 0)
            {
                SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
                SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);
                m_buffer2SwapOut[buffer] = swpOut;
                m_buffer2SwapIn[buffer] = swpIn;
            }

            m_node2ForwardSwapOut[pair.first].push_back(m_buffer2SwapOut[buffer]);
        }
    }


    // setup swapin actions
    for(auto pair : backwardSwapInNodes2matrices)
    {
        for(auto buffer : pair.second)
        {
            if(m_buffer2SwapIn.count(buffer) == 0)
            {
                SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
                SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);
                m_buffer2SwapOut[buffer] = swpOut;
                m_buffer2SwapIn[buffer] = swpIn;
            }

            m_node2BackwardSwapin[pair.first].push_back(m_buffer2SwapIn[buffer]);
        }
    }

    // setup free "actions" (this is just a resize)
    m_node2BackwardFree = lastBackwardNodes2matrices;

    assert(forwardSwapOutNodes2matrices.size() == backwardSwapInNodes2matrices.size());
    m_node2TimeStep = node2TimeStep;

    for(auto pair : node2TimeStep)
        m_timeStep2Node[pair.second] = pair.first;

    m_maxTimeStep = 0;
    for(auto pair : node2TimeStep)
        m_maxTimeStep = pair.second > m_maxTimeStep ? pair.second : m_maxTimeStep;
        

}


template<typename ElemType> void SwapManager<ElemType>::ClearActionsAndTheirMemory()
{
    cout << "Cleaning up!" << endl;
    CleanUp();

    m_buffer2SwapIn.clear();
    m_buffer2SwapOut.clear();

    m_node2ForwardSwapOut.clear();
    m_node2BackwardSwapin.clear();
    m_node2BackwardFree.clear();
}

template class SwapManager<double>;
template class SwapManager<float>;

}}}
