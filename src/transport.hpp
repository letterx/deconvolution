#ifndef _TRANSPORT_HPP_
#define _TRANSPORT_HPP_


/* Solve a transportation problem with given costs, supply and demand.
 *
 * min \sum_{i,j} c_{i,j} x_{i,j}
 * s.t. \sum_j x_{i,j} = s_i
 *      \sum_i x_{i,j} = d_j
 *      x_{i,j} <= 0
 *
 * costs is an array of size sizeSupply*sizeDemand, indexed by edge (i,j) at 
 * entry i*sizeDemand+j
 *
 * supply is array of size sizeSupply
 * demand is array of size sizeDemand
 *
 * flow is output array of same size/indexing as costs
 *
 * Must have sum of supplies = sum of demands
 */
template <typename T>
double solveTransport(int sizeSupply, int sizeDemand, const T* costs, 
        const T* supply, const T* demand, T* flow);


#endif
