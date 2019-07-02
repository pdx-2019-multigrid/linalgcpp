/*! @file */

#ifndef PARTITION_HPP__
#define PARTITION_HPP__

#include "metis.h"
#include "linalgcpp.hpp"
#include <vector>

namespace linalgcpp
{

template<typename T>
linalgcpp::Vector<int> Partition(const linalgcpp::SparseMatrix<T> adjacency, int num_parts)
{
    //TODO: Error Checking
    // - adjacency should be square
    // - num_parts should be 1 or greater
    
    int error, objval;
    int ncon = 1;
    int options[METIS_NOPTIONS] = { };
    METIS_SetDefaultOptions(options);
    
    options[METIS_OPTION_NUMBERING] = 0;
    
    int nodes = adjacency.Cols();
    std::vector<int> partitions_data(nodes);
    
    std::vector<int> indptr(adjacency.GetIndptr());
    std::vector<int> col_indices(adjacency.GetIndices());

    error = METIS_PartGraphKway(&nodes,
                                &ncon,
                                indptr.data(),
                                col_indices.data(),
                                NULL,
                                NULL,
                                NULL,
                                &num_parts,
                                NULL,
                                NULL,
                                NULL,
                                &objval,
                                partitions_data.data()
                                );
    
    linalgcpp::Vector<int> partitions(partitions_data);
    return partitions;
}

/* Right now this is assuming that there are no empty partitions / skipped integers
 *
 *
linalgcpp::SparseMatrix<double> GetInterpolationMatrix(linalgcpp::Vector<int> partitions)
{
    
    SparseMatrix<double> 
}
*/

} // namespace linalgcpp

#endif // PARTITION_HPP__
