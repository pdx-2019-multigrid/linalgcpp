#include "linalgcpp.hpp"
#include "partition.hpp"

using namespace linalgcpp;

int main()
{
    SparseMatrix<double> fine_adjacency = ReadMTXList("data/amazon_m_0.edges");
    Vector<int> partitions = Partition(fine_adjacency,20);
    
    std::cout << "Partition Vector: ";
    for (int part: partitions.data()) std::cout << part << ' ';
    std::cout << std::endl;
    
    SparseMatrix<int> interpolation = GetUnweightedInterpolator(partitions);
    SparseMatrix<double> coarse_mat = interpolation.Transpose().Mult(fine_adjacency.Mult(interpolation));

    coarse_mat.PrintDense();
}
