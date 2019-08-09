add_executable(partition_example partition_example.cpp)
target_link_libraries(partition_example partition)

function(copy_file filename)
    configure_file(
        "${PROJECT_SOURCE_DIR}/modules/partition/examples/data/${filename}"
        "${PROJECT_BINARY_DIR}/modules/partition/examples/data/${filename}" @ONLY)
endfunction(copy_file)

copy_file(amazon_m_0.edges)
copy_file(simple_graph_1.edges)
