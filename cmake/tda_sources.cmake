include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

set(TDA_SOURCES
    ${BASE_DIR}/nt/tda/old_tda/Simplex.cpp
    ${BASE_DIR}/nt/tda/old_tda/Simplex2d.cpp
    ${BASE_DIR}/nt/tda/old_tda/Shapes.cpp
    ${BASE_DIR}/nt/tda/old_tda/Basis.cpp
    ${BASE_DIR}/nt/tda/old_tda/Points.cpp
    ${BASE_DIR}/nt/tda/old_tda/Points2d.cpp
    ${BASE_DIR}/nt/tda/old_tda/KDTree.cpp
    # nt/tda/old_tda/refinement/refine.cpp <- unreleased
    ${BASE_DIR}/nt/tda/old_tda/BatchBasis.cpp
    ${BASE_DIR}/nt/tda/old_tda/BatchKDTree.cpp
    ${BASE_DIR}/nt/tda/old_tda/BatchPoints.cpp
    ${BASE_DIR}/nt/tda/BasisOverlapping.cpp
    ${BASE_DIR}/nt/tda/Boundaries.cpp
    ${BASE_DIR}/nt/tda/Homology.cpp
    ${BASE_DIR}/nt/tda/MatrixReduction.cpp
    ${BASE_DIR}/nt/tda/cpu/MatrixReduction.cpp
    ${BASE_DIR}/nt/tda/PlotDiagrams.cpp
    ${BASE_DIR}/nt/tda/Points.cpp
    ${BASE_DIR}/nt/tda/SimplexConstruct.cpp
    ${BASE_DIR}/nt/tda/SimplexRadi.cpp
    #learned persistent homology
    ${BASE_DIR}/nt/tda/nn/distance.cpp
    ${BASE_DIR}/nt/tda/nn/filtration.cpp
    ${BASE_DIR}/nt/tda/nn/laplacian.cpp
    ${BASE_DIR}/nt/tda/nn/boundaries.cpp
    ${BASE_DIR}/nt/tda/nn/loss.cpp
)


