/*! @file */

#ifndef PARTITION_HPP__
#define PARTITION_HPP__

#include "metis.h"
#include "linalgcpp.hpp"

namespace linalgcpp
{

/** @brief Wrapper to call Metis Partitioning
    @param mat graph to partition
    @param num_parts number of partitions to generate
    @param unbalance_factor allows some unbalance in partition sizes,
           where 1.0 is little unbalance and 2.0 is lots of unbalance
    @param contig generate only contiguous partitions where the partitioned subgraphs are always connected,
            requires the input graph be connected
    @param weighted use the input graph values as edge weights.
    @warning  Metis requires positive integer edge weights, so the absolute value is taken and converted to integer.
           Scale the input appropriately to obtain desired weights
    @retval partition vector with values representing the partition of the index
 */
std::vector<int> Partition(const linalgcpp::SparseMatrix<T> adjacency, int num_parts)
{
    //TODO: Error Checking
    // - adjacency should be square
    // - num_parts should be 1 or greater
    int nodes = adjacency.Cols();
    int error, objval;
    int ncon = 1;
    vector<int> partitions(nodes);

    error = METIS_PartGraphKway(&nodes,
                                &ncon,
                                adjacency.GetIndptr().data(),
                                adjacency.GetIndices().data(),
                                NULL,
                                NULL,
                                NULL,
                                num_parts,
                                NULL,
                                NULL,
                                NULL,
                                &objval,
                                partitions.data();
                                );
    return partitions;
}

linalgcpp::SparseMatrix<double> GetInterpolationMatrix(std::vector<int> partitions)
{
    SparseMatrix<double> 
}

} // namespace linalgcpp

#endif // PARTITION_HPP__
