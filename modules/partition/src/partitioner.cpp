
#include "linalgcpp.hpp"
#include "partition.hpp"
#include <string>

using namespace linalgcpp;

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << "SOURCE NUM_PARTITIONS OUTPUT" << std::endl;
    }
    std::string source, output;
    int num_parts;
    source = argv[1];
    num_parts = std::stoi(argv[2]); 
    output = argv[3];
    
    std::cout << "Loading Matrix from " << source << std::endl;
    SparseMatrix<double> adjacency = ReadMTXList(source);
    int num_nodes = adjacency.Cols();
    std::cout << "Partitioning into " << num_parts << " parts" << std::endl;
    Vector<int> partitions = Partition(adjacency,num_parts);
    std::cout << "Writing partition data to " << output << std::endl;
    
    std::ofstream parts_file(output);
    parts_file << "Id,Label,Partition" << std::endl;
    for (int i = 0; i < num_nodes; ++i)
    {
        parts_file << i+1 << ',' << i+1 << ',' << partitions[i] << std::endl;
    }
}
