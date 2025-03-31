#include <nt/Tensor.h>
#include <nt/functional/functional.h>
#include <nt/tda/Homology.h>
#include <nt/tda/PlotDiagrams.h>
#include <nt/tda/SimplexConstruct.h>
#include <nt/tda/SimplexRadi.h>
#include <nt/tda/Boundaries.h>
#include <nt/tda/MatrixReduction.h>
#include <nt/tda/nn/PH.h>
#include <nt/sparse/SparseTensor.h>


void persistent_pointcloud_2d(){
    
    int64_t dims = 2;
    // nt::Tensor cloud = nt::functional::zeros({6, 30, 30}, nt::DType::uint8);
    // nt::Tensor bools = nt::functional::randbools(cloud.shape(), 0.03); //fill 3% with 1's
    int8_t point = 1;
    // cloud[bools] = 1;
    nt::Tensor cloud({30, 30}, nt::DType::int8);
    cloud << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
             0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;

    nt::tda::PersistentHomology homologies = nt::tda::PersistentHomology::FromPointCloud(cloud, point, dims);
    
    std::cout << "dims: "<<homologies.dims() << std::endl;

    homologies.constructGroups(2); //Max is H1
    std::cout << "constructed, finding homologies..."<<std::endl;
    //finding generators and Betti numbers
    homologies.findHomology();
     
    
    //printint homology groups
    auto homology_groups = homologies.getHomologyGroups();
    int cntr = 0;
    std::cout << nt::noprintdtype << std::endl;
    for(auto& group : homology_groups){
        std::cout << "H"<<cntr<<" :"<<std::endl;
        for(auto& tup : group){
            auto [generator, birth, death] = tup;
            std::cout << "\t" << generator << ", birth: "<<birth<<", death: "<<death<<std::endl;
        }
        ++cntr;
    }

    //get a persistent diagram, barcode, and a plot of the point cloud
    nt::tda::plotPersistentDiagram(homology_groups);
    matplot::save("homology_diagram2d.png");
    matplot::show();
    nt::tda::plotBarcode(homology_groups);
    matplot::save("homology_barcode2d.png");
    matplot::show();
    nt::tda::plotPointCloud(cloud, point, dims);
    matplot::save("point_cloud2d.png");
    matplot::show();

    //find the simplex complex corresponding to the greatest amount of persistence time
    auto greatest_distance_h1 = std::max_element(homology_groups[1].begin(), homology_groups[1].end(),
                                     [](const auto& a, const auto& b){
                                        const auto& [generator_a, birth_a, death_a] = a;
                                        const auto& [generator_b, birth_b, death_b] = b;
                                        return (death_a - birth_a) < (death_b - birth_b);
                                    });
    const auto& [generator_sig, birth_sig, death_sig] = *greatest_distance_h1;
    std::cout << "significant generator: "<<generator_sig<< nt::printdtype << std::endl;
    nt::Tensor sig_complex = homologies.generatorToSimplexComplex(generator_sig, 1);
    std::cout << "significant complex: "<<sig_complex<<std::endl;
    std::cout << "birth is "<<birth_sig<<" and death is "<<death_sig<<" and the shape persists for "<<death_sig-birth_sig<<" radii"<<std::endl;
}

/*


H0 :
	Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], {30}), birth: 0, death: 0
	Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,20,24,25,26,27,28,29], {24}), birth: 0, death: 0.5
	Tensor([0,1,2,3,4,5,6,7,8,9,11,13,16,17,18,20,24,25,26,27,28,29], {22}), birth: 0, death: 0.707107
	Tensor([0,1,6,7,8,9,11,13,16,17,18,24,25,26,27,28,29], {17}), birth: 0, death: 1
	Tensor([0,7,8,9,11,13,16,17,18,25,26,27,28,29], {14}), birth: 0, death: 1.5
	Tensor([0,7,8,9,11,13,16,26,27,28,29], {11}), birth: 0, death: 1.80278
	Tensor([0,7,8,9,11,13,16,29], {8}), birth: 0, death: 2
	Tensor([0,7,9,11,13], {5}), birth: 0, death: 2.23607
	Tensor([9,11,13], {3}), birth: 0, death: 2.5
	Tensor (Null), birth: 0, death: 3.60555
H1 :
	Tensor([1,3,4,8,9,11,12,15,17,19,21,22,23,24,31,32,37,38,42,53,59,68,107], {23}), birth: 5.59017, death: 6.67083
	Tensor([31,61,67], {3}), birth: 4, death: 4.03113
	Tensor([4,8,9,11,12,17,19,21,22,24,29,31,37,38,53,59,61,68,90], {19}), birth: 4.71699, death: 6.67083
	Tensor([3,23,36], {3}), birth: 2.69258, death: 2.82843
	Tensor([11,31,62], {3}), birth: 3.64005, death: 3.80789
	Tensor([8,11,17,22,31,37,59,61,64], {9}), birth: 3.90512, death: 5.5
	Tensor([13,29,53,63], {4}), birth: 3.64005, death: 3.80789
	Tensor([0,12,13], {3}), birth: 1.58114, death: 1.80278
Press ENTER to continue...
Warning: empty x range [0:0], adjusting to [-1:1]
Warning: empty y range [0.1:0.1], adjusting to [0.099:0.101]
Warning: empty x range [0:0], adjusting to [-1:1]
Warning: empty y range [0.1:0.1], adjusting to [0.099:0.101]
Warning: empty x range [0:0], adjusting to [-1:1]
Warning: empty y range [0.1:0.1], adjusting to [0.099:0.101]
Press ENTER to continue...
Press ENTER to continue...
significant generator: Tensor([4,8,9,11,12,17,19,21,22,24,29,31,37,38,53,59,61,68,90], {19})
significant complex: Tensor([[[23,2],
          [23,4]]


         [[1,5],
          [3,4]]


         [[22,4],
          [23,2]]


         [[3,8],
          [3,11]]


         [[24,17],
          [24,20]]


         [[1,5],
          [3,2]]


         [[26,8],
          [26,12]]


         [[23,4],
          [27,4]]


         [[3,4],
          [3,8]]


         [[26,8],
          [27,4]]


         [[16,19],
          [18,23]]


         [[3,11],
          [5,15]]


         [[3,2],
          [8,0]]


         [[24,17],
          [26,12]]


         [[18,23],
          [24,20]]


         [[8,0],
          [14,4]]


         [[5,15],
          [11,11]]


         [[14,4],
          [22,4]]


         [[11,11],
          [16,19]]
], {19,2,2})

DTypeInteger64
birth is 4.71699 and death is 6.67083 and the shape persists for 1.95384 radii

*/


void persistent_pointcloud_3d(){
    
    int64_t dims = 3;
    nt::Tensor cloud = nt::functional::zeros({1, 30, 30, 30}, nt::DType::int8);
    //I want it to have 50 random points so:
    double percent = 50.0 / (30.0 * 30.0 * 30.0);
    std::cout << "percent is "<<percent<<std::endl;
    nt::Tensor bools = nt::functional::randbools(cloud.shape(), percent); //fill 0.3% with 1's
    int8_t point = 1;
    cloud[bools] = 1;
    std::cout << "there are "<<nt::functional::count(bools) << "points"<<std::endl;

    nt::tda::PersistentHomology homologies = nt::tda::PersistentHomology::FromPointCloud(cloud, point, dims);
    
    std::cout << "constructing groups"<<std::endl;
    homologies.constructGroups(3); //Max is H2
    std::cout << "constructed, finding homologies..."<<std::endl;
    //finding generators and Betti numbers
    homologies.findHomology(3);
     
    
    //printint homology groups
    auto homology_groups = homologies.getHomologyGroups();
    int cntr = 0;
    std::cout << nt::noprintdtype << std::endl;
    for(auto& group : homology_groups){
        std::cout << "H"<<cntr<<" :"<<std::endl;
        for(auto& tup : group){
            auto [generator, birth, death] = tup;
            std::cout << "\t" << generator << ", birth: "<<birth<<", death: "<<death<<std::endl;
        }
        ++cntr;
    }

    //get a persistent diagram, barcode, and a plot of the point cloud
    nt::tda::plotPersistentDiagram(homology_groups);
    matplot::save("homology_diagram3d.png");
    matplot::show();
    nt::tda::plotBarcode(homology_groups);
    matplot::save("homology_barcode3d.png");
    matplot::show();
    nt::tda::plotPointCloud(cloud, point, dims);
    matplot::save("point_cloud3d.png");
    matplot::show();

    //find the simplex complex corresponding to the greatest amount of persistence time
    auto greatest_distance_h1 = std::max_element(homology_groups[1].begin(), homology_groups[1].end(),
                                     [](const auto& a, const auto& b){
                                        const auto& [generator_a, birth_a, death_a] = a;
                                        const auto& [generator_b, birth_b, death_b] = b;
                                        return (death_a - birth_a) < (death_b - birth_b);
                                    });
    const auto& [generator_sig, birth_sig, death_sig] = *greatest_distance_h1;
    std::cout << "significant generator: "<<generator_sig<< nt::printdtype << std::endl;
    nt::Tensor sig_complex = homologies.generatorToSimplexComplex(generator_sig, 1);
    std::cout << "significant complex: "<<sig_complex<<std::endl;
    std::cout << "birth is "<<birth_sig<<" and death is "<<death_sig<<" and the shape persists for "<<death_sig-birth_sig<<" radii"<<std::endl;

    //find the simplex complex corresponding to the greatest amount of persistence time
    auto greatest_distance_h2 = std::max_element(homology_groups[2].begin(), homology_groups[2].end(),
                                     [](const auto& a, const auto& b){
                                        const auto& [generator_a, birth_a, death_a] = a;
                                        const auto& [generator_b, birth_b, death_b] = b;
                                        return (death_a - birth_a) < (death_b - birth_b);
                                    });
    const auto& [generator_sig_2, birth_sig_2, death_sig_2] = *greatest_distance_h2;
    std::cout << "significant generator: "<<generator_sig_2<< nt::printdtype << std::endl;
    nt::Tensor sig_complex_2 = homologies.generatorToSimplexComplex(generator_sig_2, 2);
    std::cout << "significant complex: "<<sig_complex_2<<std::endl;
    std::cout << "birth is "<<birth_sig_2<<" and death is "<<death_sig_2<<" and the shape persists for "<<death_sig_2-birth_sig_2<<" radii"<<std::endl;
    
}

//example output:
/*
H0 :
	Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], {30}), birth: 0, death: 0
	Tensor([0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], {28}), birth: 0, death: 1.22474
	Tensor([0,1,2,3,4,5,8,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,27,28,29], {26}), birth: 0, death: 1.5
	Tensor([0,1,2,3,4,5,9,10,11,13,14,17,19,20,21,22,23,24,25,26,27,28,29], {23}), birth: 0, death: 2.5
	Tensor([0,1,2,3,4,5,9,10,11,13,14,17,19,21,22,24,25,26,28,29], {20}), birth: 0, death: 2.59808
	Tensor([0,2,3,5,9,11,13,14,17,19,21,22,24,25,26,28,29], {17}), birth: 0, death: 2.69258
	Tensor([0,2,3,5,11,14,17,19,22,24,25,26,28,29], {14}), birth: 0, death: 3.20156
	Tensor([0,2,3,5,11,14,17,22,24,26,29], {11}), birth: 0, death: 3.3541
	Tensor([0,2,3,5,11,14,22,24,26], {9}), birth: 0, death: 3.5
	Tensor([2,3,5,11,14,22,24], {7}), birth: 0, death: 4.15331
	Tensor([2,3,11,14,22], {5}), birth: 0, death: 4.55522
	Tensor([3,14,22], {3}), birth: 0, death: 4.71699
	Tensor (Null), birth: 0, death: 4.7697
H1 :
	Tensor([5,9,21,28,86], {5}), birth: 6.87386, death: 6.96419
	Tensor([5,24,28,31,32,33,36,38,72], {9}), birth: 6.44205, death: 7.82624
	Tensor([5,6,12,13,14,15,20,21,22,24,28,31,36,85], {14}), birth: 6.87386, death: 8.45577
	Tensor([12,13,15,22,35,51,57], {7}), birth: 5.78792, death: 6.20484
	Tensor([5,12,13,14,15,21,22,24,28,31,33,35,36,38,41], {15}), birth: 5.61249, death: 8.544
	Tensor([5,10,12,13,14,15,17,21,24,28,31,33,35,36,38,41], {16}), birth: 5.12348, death: 5.52268
	Tensor([0,23,34], {3}), birth: 4.71699, death: 4.74342
	Tensor([8,10,15,16,17,19,29], {7}), birth: 4.74342, death: 5.52268
	Tensor([2,15,16,19,20,22], {6}), birth: 5.61249, death: 6.20484
	Tensor([0,20,29], {3}), birth: 4.55522, death: 4.63681
	Tensor([33,38,101], {3}), birth: 7.51665, death: 7.68115
	Tensor([8,10,15,16,17,19,20,23,34], {9}), birth: 4.74342, death: 4.94975
	Tensor([10,17,22], {3}), birth: 4.15331, death: 4.1833
	Tensor([0,8,10,15,16,17,19,20], {8}), birth: 4.55522, death: 4.74342
	Tensor([2,10,15,16,17,19,20], {7}), birth: 4.09268, death: 4.55522
	Tensor([5,9,11], {3}), birth: 3.20156, death: 3.77492
H2 :
	Tensor([9,107,136,138], {4}), birth: 7.71362, death: 7.79423
	Tensor([1,33,103,104], {4}), birth: 7.15891, death: 7.22842
	Tensor([16,131,132,133], {4}), birth: 7.68115, death: 7.6974
	Tensor([5,42,53,54], {4}), birth: 6.40312, death: 6.42262
	Tensor([0,8,39,40], {4}), birth: 6.10328, death: 6.20484
	Tensor([9,13,98,99], {4}), birth: 7.03562, death: 7.1239
	Tensor([19,20,29,30], {4}), birth: 5.72276, death: 5.78792
	Tensor([3,18,48,49], {4}), birth: 6.26498, death: 6.36396
	Tensor([5,19,34,35], {4}), birth: 5.87367, death: 5.89491
	Tensor([1,14,21,22], {4}), birth: 5.47723, death: 5.61249
Press ENTER to continue...qt.pointer.dispatch: skipping QEventPoint(id=1 ts=0 pos=0,0 scn=1108.75,494.145 gbl=1108.75,494.145 Released ellipse=(1x1 ∡ 0) vel=0,0 press=-1108.75,-494.145 last=-1108.75,-494.145 Δ 1108.75,494.145) : no target window
qt.pointer.dispatch: skipping QEventPoint(id=2 ts=0 pos=0,0 scn=869.528,312.347 gbl=869.528,312.347 Released ellipse=(1x1 ∡ 0) vel=0,0 press=-869.528,-312.347 last=-869.528,-312.347 Δ 869.528,312.347) : no target window
qt.pointer.dispatch: skipping QEventPoint(id=2 ts=0 pos=0,0 scn=891.324,16.1303 gbl=891.324,16.1303 Released ellipse=(1x1 ∡ 0) vel=0,0 press=-891.324,-16.1303 last=-891.324,-16.1303 Δ 891.324,16.1303) : no target window
qt.pointer.dispatch: skipping QEventPoint(id=3 ts=0 pos=0,0 scn=1180.04,178.671 gbl=1180.04,178.671 Released ellipse=(1x1 ∡ 0) vel=0,0 press=-1180.04,-178.671 last=-1180.04,-178.671 Δ 1180.04,178.671) : no target window

Warning: empty x range [0:0], adjusting to [-1:1]
Warning: empty y range [0.1:0.1], adjusting to [0.099:0.101]
Warning: empty x range [0:0], adjusting to [-1:1]
Warning: empty y range [0.1:0.1], adjusting to [0.099:0.101]
Warning: empty x range [0:0], adjusting to [-1:1]
Warning: empty y range [0.1:0.1], adjusting to [0.099:0.101]
Press ENTER to continue...
Press ENTER to continue...
significant generator: Tensor([5,12,13,14,15,21,22,24,28,31,33,35,36,38,41], {15})
significant complex: Tensor([[[22,29,4],
          [27,28,5]]


         [[26,10,15],
          [28,14,20]]


         [[22,5,13],
          [26,10,15]]


         [[28,14,20],
          [29,20,17]]


         [[19,7,19],
          [22,5,13]]


         [[26,27,14],
          [29,20,17]]


         [[11,9,20],
          [19,7,19]]


         [[2,17,8],
          [5,23,3]]


         [[26,27,14],
          [27,28,5]]


         [[13,28,2],
          [22,29,4]]


         [[2,17,8],
          [2,25,13]]


         [[11,9,20],
          [16,17,21]]


         [[5,23,3],
          [13,28,2]]


         [[2,25,13],
          [11,25,17]]


         [[11,25,17],
          [16,17,21]]
], {15,2,3})

DTypeInteger64
birth is 5.61249 and death is 8.544 and the shape persists for 2.93152 radii
significant generator: Tensor([1,14,21,22], {4})

DTypeInteger64
significant complex: Tensor([[[22,29,4],
          [23,23,5],
          [27,28,5]]


         [[23,23,5],
          [26,27,14],
          [27,28,5]]


         [[22,29,4],
          [23,23,5],
          [26,27,14]]


         [[22,29,4],
          [26,27,14],
          [27,28,5]]
], {4,3,3})

DTypeInteger64
birth is 5.47723 and death is 5.61249 and the shape persists for 0.135261 radii
*/



std::vector<std::tuple<nt::Tensor, double, double> > getHomologyGrouping(nt::Tensor& cloud, int64_t& dims, int8_t point){
    nt::tda::PersistentHomology homologies = nt::tda::PersistentHomology::FromPointCloud(cloud, point, dims);
    homologies.constructGroups(2); //Max is H1
    homologies.findHomology();
    auto homology_groups = homologies.getHomologyGroups();
    return homology_groups[1]; //getting H1
}

std::vector<std::tuple<nt::Tensor, double, double> > getHomologyGrouping(nt::Tensor& cloud, int64_t& dims, int8_t point, nt::Tensor weight){
    nt::tda::PersistentHomology homologies = nt::tda::PersistentHomology::FromPointCloud(cloud, point, dims);
    homologies.add_weight(weight);
    homologies.constructGroups(2); //Max is H1
    homologies.findHomology();
    auto homology_groups = homologies.getHomologyGroups();
    return homology_groups[1]; //getting H1
}


void print_homology_group(std::vector<std::tuple<nt::Tensor, double, double> >& group){
    for(auto& tup : group){
        auto [generator, birth, death] = tup;
        std::cout << "\t" << generator << ", birth: "<<birth<<", death: "<<death<<std::endl;
    }
}


void round_tensor(nt::Tensor& t){
    if(t.dtype == nt::DType::Float16){
        t.arr_void().execute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Float16> > >
        ([](auto begin, auto end){
            std::transform(begin, end, begin, [](nt::float16_t x){
                return _NT_FLOAT32_TO_FLOAT16_(std::round(_NT_FLOAT16_TO_FLOAT32_(x)));
            });
        });
        return;
    }
#ifdef _128_FLOAT_SUPPORT_
    if(t.dtype == nt::DType::Float128){
        t.arr_void().execute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Float128> > >
        ([](auto begin, auto end){
            std::transform(begin, end, begin, [](nt::float128_t x){return std::round(x);});
        });
        return;
    }
#endif
    if(t.dtype == nt::DType::Float32 || t.dtype == nt::DType::Float64){
        t.arr_void().execute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Float32, nt::DType::Float64> > >
        ([](auto begin, auto end){
            std::transform(begin, end, begin, [](auto x){return std::round(x);});
        });
        return;
    }
    if(t.dtype == nt::DType::Complex64 || t.dtype == nt::DType::Complex128){
        t.arr_void().execute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Complex64, nt::DType::Complex128> > >
        ([](auto begin, auto end){
            // using type = nt::utils::ItteratorBaseType_t<decltype(begin)>;
            std::transform(begin, end, begin, [](auto x){
                    return decltype(x)(std::round(x.real()), std::round(x.imag()));
            });
        });
    }
    if(t.dtype == nt::DType::Complex32){
        t.arr_void().execute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Complex32> > >
        ([](auto begin, auto end){
            std::transform(begin, end, begin, [](auto x){
                    return nt::complex_32(
                        _NT_FLOAT32_TO_FLOAT16_(std::round(_NT_FLOAT16_TO_FLOAT32_(x.real()))), 
                        _NT_FLOAT32_TO_FLOAT16_(std::round(_NT_FLOAT16_TO_FLOAT32_(x.imag()))));
            });
        });
    }
}

void persistent_gradient(){
    
    int64_t dims = 2;
    // nt::Tensor cloud = nt::functional::zeros({6, 30, 30}, nt::DType::uint8);
    // nt::Tensor bools = nt::functional::randbools(cloud.shape(), 0.03); //fill 3% with 1's
    int8_t point = 1;
    // cloud[bools] = 1;
    nt::Tensor cloud({9, 9}, nt::DType::Float32);
    cloud << 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0;

        
    
    std::cout << nt::noprintdtype;
    nt::Tensor points_w = nt::functional::where(cloud >= 1);
    nt::Tensor points_s = nt::functional::stack(points_w).transpose(-1, -2).to(nt::DType::Float32);
    std::cout << "points_s shape: "<< points_s.shape() << std::endl;
    int64_t D = points_s.shape()[-1];
    nt::Tensor dif = points_s.view(-1, 1, D) - points_s.view(1, -1, D);
    nt::Tensor dist_matrix = nt::functional::sqrt(dif.pow(2).sum(-1));
    std::cout << "distance matrix: "<<std::endl<<dist_matrix<<std::endl;

    nt::Tensor delta_dist_matrix = nt::functional::rand(0, 1.8, dist_matrix.shape(), nt::DType::Float32);
    std::cout << "distance matrix gradient: "<<delta_dist_matrix<<std::endl;
    
    //let j and i represent vertexes in the cloud
    //such that j = (x1, y1) and i = (x2, y2)
    //dist_matrix[i][j] = sqrt(cloud[i] - cloud[j])
    //delta_dist_matrix is the gradient of the dist_matrix
    //
    //g[i], dist_matrix[i], delta_dist_matrix[i]  is the index of the point i
    //now to find the gradient with respect to each point
    //g[i] represents the gradient of a point at i
    //g[i] = [sum((x[i] - x[j]) / dist_matrix[i][j]) | i != j] * delta_dist_matrix[i][j]
    //this would be the same thing as:
    //summed_dif = dif.sum(-1)
    //dist_matrix.fill_diagonal_(1.0); //summed_dif[i] should be 0 anyways
    //g[i] = (summed_dif[i] * delta_dist_matrix[i]) / dist_matrix.sum(-1)
    //
    //so to get the gradients of the points in a shape of {N, D}:
    int64_t N = dist_matrix.shape()[0];
    nt::Tensor G_a = dif * delta_dist_matrix.view(N, N, 1);
    //shape {N, N, D}
    dist_matrix.fill_diagonal_(1.0);
    nt::Tensor G_b = G_a / dist_matrix.view(N, N, 1);
    //shape: {N, N, D}
    nt::Tensor Gd = G_b.sum(-2); 
    //this is now the gradient of the points
    std::cout << nt::printdtype << std::endl;
    std::cout << "Gd: "<<Gd<<std::endl;
        
    std::cout << "Points: "<<points_s<<std::endl;

    nt::Scalar lr = 0.7;
    nt::Tensor updated = nt::functional::relu(points_s - (Gd * lr));
    round_tensor(updated);
    std::cout << updated << std::endl;
    nt::Tensor cloud_grad = nt::functional::zeros_like(cloud);
    nt::Tensor old = points_s.transpose(-1, -2).to(nt::DType::int64).split_axis(-2);
    cloud_grad[old] = -1;
    cloud_grad[updated.transpose(-1, -2).to(nt::DType::int64).split_axis(-2)] = 1;
    std::cout << cloud_grad<< std::endl << cloud << std::endl;
    std::cout << cloud + cloud_grad << std::endl;
    //now has shape {N, D}
    //and it can be used for point addition or removal
    //dist_matrix and delta_dist_matrix have shape {N, N}


    // auto no_weight = getHomologyGrouping(cloud, dims, point);
    // std::cout << "H1 No Weight:" << std::endl;
    // print_homology_group(no_weight);

    // nt::Tensor weight = nt::functional::rand(1, 3, {6, 6}, nt::DType::Float64);
    // auto ne_weight = getHomologyGrouping(cloud, dims, point, weight);
    // std::cout << "H1 No Epsilon Weight:" << std::endl;
    // print_homology_group(ne_weight);

    // nt::Tensor epsilon = nt::functional::zeros({6, 6}, nt::DType::Float64);
    // epsilon[3][2] = 1;
    // auto m_epsilon = getHomologyGrouping(cloud, dims, point, weight - epsilon);
    // std::cout << "H1 Minus Epsilon Weight:" << std::endl;
    // print_homology_group(m_epsilon);

    // auto p_epsilon = getHomologyGrouping(cloud, dims, point, weight + epsilon);
    // std::cout << "H1 Plus Epsilon Weight:" << std::endl;
    // print_homology_group(p_epsilon);
    
    // std::cout << "Weight: "<<weight<<std::endl;

}

void persistent_dist_mat_gradient(){
    
    int64_t dims = 2;
    // nt::Tensor cloud = nt::functional::zeros({6, 30, 30}, nt::DType::uint8);
    // nt::Tensor bools = nt::functional::randbools(cloud.shape(), 0.03); //fill 3% with 1's
    int8_t point = 1;
    // cloud[bools] = 1;
    nt::TensorGrad cloud(nt::Tensor({9, 9}, nt::DType::Float32));
    cloud << 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0;

    
    nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
    std::cout << "dist mat: "<<dist_mat<<std::endl;
    nt::Tensor wanted = nt::functional::rand(0.3, 2.5, dist_mat.shape(), dist_mat.dtype);
    nt::Tensor grad = dist_mat.tensor - wanted;
    dist_mat.backward(grad);
    std::cout << "cloud: " << cloud << std::endl;
    std::cout << "cloud grad: " << cloud.grad_value() << std::endl;
    cloud.update();
    std::cout << "cloud: " << cloud << std::endl;

}

//tests gettingg the gradient of a simplex complex
void persistent_simplex_complex_gradient(){
    nt::TensorGrad cloud(nt::Tensor({9, 9}, nt::DType::Float32));
    cloud << 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0;
    nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
    auto [simplex_complex, radi] = nt::tda::VRfiltration(dist_mat, 3);
    std::cout <<"radi: "<< radi << std::endl;
    std::cout << "simplex complex: "<<simplex_complex<<std::endl;
    //negative values increase the radii
    //positive values decrease the radii
    nt::Tensor wanted = nt::functional::rand(1.0, 6.0, radi.shape(), nt::DType::Float32);
    nt::TensorGrad loss = nt::tda::loss::filtration_loss(radi, wanted);
    loss.backward();
    std::cout << "cloud: " << cloud << std::endl;
    std::cout << "cloud grad: " << cloud.grad_value() << std::endl;
    cloud.update();
    std::cout << "cloud: " << cloud << std::endl;
    std::cout << "wanted: "<<wanted<<std::endl;
    std::cout << "loss: "<<loss<<std::endl;
    dist_mat = nt::tda::cloudToDist(cloud, 1);
    auto [n_simplex_complex, n_radi] = nt::tda::VRfiltration(dist_mat, 3);
    std::cout << "n_radi: "<<n_radi<<std::endl;
    std::cout << "n_simplex_complex: "<<n_simplex_complex<<std::endl;
}



void unit_simultaneous_test(int64_t increment){
    // nt::Tensor cloud = nt::tda::generate_random_cloud({30, 30});
    nt::Tensor cloud({30, 30}, nt::DType::int8);
    cloud << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
             0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    nt::Tensor points = nt::tda::extract_points_from_cloud(cloud, 1, 2);
    nt::tda::BasisOverlapping balls(points);
    
    auto [simplex_1, radii_1] = nt::get<2>(nt::tda::find_all_simplicies(1, points, balls, true));
    auto [simplex_2, radii_2] = nt::get<2>(nt::tda::find_all_simplicies(2, points, balls, true));
    auto [simplex_3, radii_3] = nt::get<2>(nt::tda::find_all_simplicies(3, points, balls, true));
    nt::SparseTensor Boundary_K = nt::tda::compute_boundary_matrix_index(simplex_2.item<nt::Tensor>(),
                                                                simplex_1.item<nt::Tensor>());
    nt::SparseTensor Boundary_Kp1 = nt::tda::compute_boundary_matrix_index(simplex_3.item<nt::Tensor>(),
                                                                simplex_2.item<nt::Tensor>());
    std::cout << "boundary k: "<<Boundary_K.shape()<<std::endl;
    std::cout << "boundary k+1: "<<Boundary_Kp1.shape()<<std::endl;
    
    int64_t start_rows1 = 0;
    int64_t end_rows1 = 0;
    int64_t start_cols1 = 0;
    int64_t end_cols1 = 0;

    int64_t k1start_rows1 = 0;
    int64_t k1end_rows1 = 0;
    int64_t k1start_cols1 = 0;
    int64_t k1end_cols1 = 0;

    nt::Tensor BK = Boundary_K.underlying_tensor().transpose(-1, -2).to(nt::DType::Float32);
    nt::Tensor BK_1 = Boundary_Kp1.underlying_tensor().to(nt::DType::Float32);
    nt::Tensor A = BK.clone();
    nt::Tensor B = BK_1.clone();
    std::cout << "A shape: "<<A.shape()<<", B shape: "<<B.shape()<<std::endl;
    while(true){
        end_rows1 = std::min(end_rows1+increment, Boundary_K.shape()[0]);
        end_cols1 = std::min(end_cols1+increment, Boundary_K.shape()[1]);
        k1end_rows1 = end_cols1;
        k1end_cols1 = std::min(k1end_cols1+increment, Boundary_Kp1.shape()[1]);
        if(k1end_cols1 == Boundary_Kp1.shape()[1]) break;
        nt::tda::partialColReduce(A, start_cols1, start_rows1, end_cols1, end_rows1);
        nt::tda::partialRowReduce(B, k1start_rows1, k1start_cols1, k1end_rows1, k1end_cols1);

        nt::SparseTensor _sub_bk = Boundary_K[{nt::my_range(0, end_rows1), nt::my_range(0, end_cols1)}];
        nt::SparseTensor _sub_bk1 = Boundary_Kp1[{nt::my_range(0, k1end_rows1), nt::my_range(0, k1end_cols1)}];
        auto [_sub_A, _sub_B] = nt::get<2>(nt::tda::simultaneousReduce(_sub_bk, _sub_bk1));
        nt::tda::finishRowReducing(_sub_B);
        
        nt::Tensor new_A = A[{nt::my_range(0, end_cols1), nt::my_range(0, end_rows1)}];
        nt::Tensor new_B = B[{nt::my_range(0, k1end_rows1), nt::my_range(0, k1end_cols1)}];
        nt::Tensor equal_A = (new_A == _sub_A);
        nt::Tensor equal_B = (new_B == _sub_B);
        bool equal_a = nt::functional::all(equal_A);
        bool equal_b = nt::functional::all(equal_B);
        //the important thing is that these match up
        std::cout << "ranks: {"<<nt::tda::numPivotRows(new_A)<<','<<nt::tda::numPivotRows(new_B)<<"}, {"
                    <<nt::tda::numPivotRows(_sub_A) << ',' << nt::tda::numPivotRows(_sub_B) << "}" << std::endl;
        // if(equal_a == false){
        //     std::cout <<nt::noprintdtype << new_A<<','<<_sub_A<<','<<equal_A<<std::endl << nt::printdtype;
        // }
        std::cout << std::boolalpha << nt::functional::all(equal_A)
                         << " " << nt::functional::all(equal_B)
              << " {"
              << end_rows1<<','<<end_cols1<<','<<k1end_cols1<<"}" << std::endl;; // so far when starting at 0 the algorithm works ! :))

        start_rows1 = 0; //remains 0
        start_cols1 = end_cols1; //increments
        k1start_rows1 = 0; //increments
        k1start_cols1 = k1end_cols1; //remains
        if(start_cols1 == Boundary_K.shape()[1]) {start_cols1 = Boundary_K.shape()[1]-(increment*2);}
    }

}


void better_simultaneous_reduce_test(){
    //NOTE:
    //  In order for this to work, the start rows of A always has to be zero
    //  and the start rows of B always has to be 0
    //
    //NOTE 2:
    //  kend_cols == k1end_rows  [ALWAYS]
    //  if not, this is an invalid matrix to reduce
    //  
    //How it works:
    //  partialColReduce:
    //      It takes a transpose of Boundary_K
    //          - this makes it faster
    //      A = Boundary_K transpose
    //      it then takes the start_rows, start_cols, end_rows, end_cols of A
    //      it then changes the memory of A to be reduced within those bounds
    //
    //  partialRowReduce:
    //      B = Boundary_K+1 as floats
    //      it then takes the start_rows, start_cols, end_rows, end_cols of B
    //      it then changes the memory of B to be reduced within those bounds
    //
    //
    //As long as the above is followed properly, the boundary matrices can be reduced
    //  incrementally making the calculation of betti numbers faster
    //  and the extraction of when betti numbers are not 0 faster
    nt::Tensor cloud = nt::tda::generate_random_cloud({30, 30});
    nt::Tensor points = nt::tda::extract_points_from_cloud(cloud, 1, 2);
    nt::tda::BasisOverlapping balls(points);
    
    auto [simplex_1, radii_1] = nt::get<2>(nt::tda::find_all_simplicies(1, points, balls, true));
    auto [simplex_2, radii_2] = nt::get<2>(nt::tda::find_all_simplicies(2, points, balls, true));
    auto [simplex_3, radii_3] = nt::get<2>(nt::tda::find_all_simplicies(3, points, balls, true));
    nt::SparseTensor Boundary_K = nt::tda::compute_boundary_matrix_index(simplex_2.item<nt::Tensor>(),
                                                                simplex_1.item<nt::Tensor>());
    nt::SparseTensor Boundary_Kp1 = nt::tda::compute_boundary_matrix_index(simplex_3.item<nt::Tensor>(),
                                                                simplex_2.item<nt::Tensor>());
    std::cout << "boundary k: "<<Boundary_K.shape()<<std::endl;
    std::cout << "boundary k+1: "<<Boundary_Kp1.shape()<<std::endl;
    
    int64_t start_rows1 = 0;
    int64_t end_rows1 = Boundary_K.shape()[0];
    int64_t start_cols1 = 0;
    int64_t end_cols1 = 40;

    int64_t k1start_rows1 = 0;
    int64_t k1end_rows1 = end_cols1;
    int64_t k1start_cols1 = 0;
    int64_t k1end_cols1 = 54;

    int64_t start_rows2 = 0; //remains 0
    int64_t end_rows2 = Boundary_K.shape()[0];
    int64_t start_cols2 = end_cols1;
    int64_t end_cols2 = start_cols2 + 30;

    int64_t k1start_rows2 = 0;
    int64_t k1end_rows2 = k1end_rows1 + 30;
    int64_t k1start_cols2 = k1end_cols1;
    int64_t k1end_cols2 = k1end_cols1 + 5;

    nt::Tensor BK = Boundary_K.underlying_tensor().transpose(-1, -2).to(nt::DType::Float32);
    nt::Tensor BK_1 = Boundary_Kp1.underlying_tensor().to(nt::DType::Float32);
    nt::Tensor A = BK.clone();
    nt::Tensor B = BK_1.clone();
    std::cout << "A shape: "<<A.shape()<<", B shape: "<<B.shape()<<std::endl;
    nt::tda::partialColReduce(A, start_cols1, start_rows1, end_cols1, end_rows1);
    nt::tda::partialRowReduce(B, k1start_rows1, k1start_cols1, k1end_rows1, k1end_cols1);

    nt::SparseTensor _sub_bk = Boundary_K[{nt::my_range(0, end_rows1), nt::my_range(0, end_cols1)}];
    nt::SparseTensor _sub_bk1 = Boundary_Kp1[{nt::my_range(0, k1end_rows1), nt::my_range(0, k1end_cols1)}];
    auto [_sub_A, _sub_B] = nt::get<2>(nt::tda::simultaneousReduce(_sub_bk, _sub_bk1));
    nt::tda::finishRowReducing(_sub_B);

    nt::Tensor new_A = A[{nt::my_range(0, end_cols1), nt::my_range(0, end_rows1)}];
    nt::Tensor new_B = B[{nt::my_range(0, k1end_rows1), nt::my_range(0, k1end_cols1)}];
    nt::Tensor equal_A = (new_A == _sub_A);
    nt::Tensor equal_B = (new_B == _sub_B);
    std::cout << std::boolalpha << nt::functional::all(equal_A)
                         << " " << nt::functional::all(equal_B)
              << std::noboolalpha << std::endl; // so far when starting at 0 the algorithm works ! :))

    
    nt::tda::partialColReduce(A, start_cols2, start_rows2, end_cols2, end_rows2);
    nt::tda::partialRowReduce(B, k1start_rows2, k1start_cols2, k1end_rows2, k1end_cols2);

    nt::SparseTensor _2_sub_bk = Boundary_K[{nt::my_range(0, end_rows2), nt::my_range(0, end_cols2)}];
    nt::SparseTensor _2_sub_bk1 = Boundary_Kp1[{nt::my_range(0, k1end_rows2), nt::my_range(0, k1end_cols2)}];
    auto [_2_sub_A, _2_sub_B] = nt::get<2>(nt::tda::simultaneousReduce(_2_sub_bk, _2_sub_bk1));
    nt::tda::finishRowReducing(_2_sub_B);


    nt::Tensor new_A_2 = A[{nt::my_range(0, end_cols2), nt::my_range(0, end_rows2)}];
    nt::Tensor new_B_2 = B[{nt::my_range(0, k1end_rows2), nt::my_range(0, k1end_cols2)}];
    nt::Tensor equal_A_2 = (new_A_2 == _2_sub_A);
    nt::Tensor equal_B_2 = (new_B_2 == _2_sub_B); 
    std::cout << std::boolalpha << nt::functional::all(equal_A_2)
                         << " " << nt::functional::all(equal_B_2)
                         // << " " << nt::functional::all(interm_sub == interm_update[{nt::my_range(0, k1end_rows2), nt::my_range(0, k1end_cols2)}])
              << std::noboolalpha << std::endl; 
}


std::map<double,
    std::tuple<int64_t, int64_t, int64_t>
    > construct_boundary_radi_map(
        const std::map<double, std::array<int64_t, 2>>& map_a,
        const std::map<double, std::array<int64_t, 2>>& map_b){
    auto begin_a = map_a.begin();
    auto end_a = map_a.end();
    auto begin_b = map_b.begin();
    auto end_b = map_b.end();
    std::map<double,
        std::tuple<int64_t, int64_t, int64_t>
        > out_map;
    //iterate until none of the indices are 0
    while(begin_a != end_a && (begin_a->second[0] == 0 || begin_a->second[1] == 0)) ++begin_a;
    while(begin_b != end_b && (begin_b->second[0] == 0 || begin_b->second[1] == 0)) ++begin_b;
    if(begin_a == end_a || begin_b == end_b){
        return out_map;
    }
    for(;begin_a != end_a; ++begin_a){
        while(begin_b != end_b && begin_b->second[0] < begin_a->second[1]) ++begin_b;
        if(begin_a->second[1] < begin_b->second[0]) continue;
        while(begin_b != end_b && begin_b->second[0] == begin_a->second[1] ){
            out_map[begin_b->first] = 
                std::make_tuple(begin_a->second[0], begin_a->second[1], begin_b->second[1]);
            ++begin_b;
        }
    }
    return out_map;
}

void simultaneous_betti_number_test(){
    nt::Tensor cloud({30, 30}, nt::DType::int8);
    cloud << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
             0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    nt::Tensor points = nt::tda::extract_points_from_cloud(cloud, 1, 2);
    nt::tda::BasisOverlapping balls(points);
    
    auto [simplex_1, radii_1] = nt::get<2>(nt::tda::find_all_simplicies(1, points, balls, true));
    auto [simplex_2, radii_2] = nt::get<2>(nt::tda::find_all_simplicies(2, points, balls, true));
    auto [simplex_3, radii_3] = nt::get<2>(nt::tda::find_all_simplicies(3, points, balls, true));
    nt::SparseTensor Boundary_K = nt::tda::compute_boundary_matrix_index(simplex_2.item<nt::Tensor>(),
                                                                simplex_1.item<nt::Tensor>());
    nt::SparseTensor Boundary_Kp1 = nt::tda::compute_boundary_matrix_index(simplex_3.item<nt::Tensor>(),
                                                                simplex_2.item<nt::Tensor>());
    std::cout << "boundary k: "<<Boundary_K.shape()<<std::endl;
    std::cout << "boundary k+1: "<<Boundary_Kp1.shape()<<std::endl;
    
    std::set<double> rSimplex1 = nt::tda::get_radi_set(radii_1);
    std::set<double> rSimplex2 = nt::tda::get_radi_set(radii_2);
    std::set<double> rSimplex3 = nt::tda::get_radi_set(radii_3);
    
    std::map<double, std::array<int64_t, 2>> sigma_map_a = 
        nt::tda::make_simplex_radi_map(
            Boundary_K.shape()[0],
            radii_2[0].item<nt::Tensor>());
    std::map<double, std::array<int64_t, 2>> sigma_map_b = 
        nt::tda::make_simplex_radi_map(
            radii_2[0].item<nt::Tensor>(),
            radii_3[0].item<nt::Tensor>());
    std::cout << "sigma a map:"<<std::endl;
    for(const auto& val : sigma_map_a){
        std::cout << val.first<<": {"<<val.second[0]<<','<<val.second[1]<<"} ";
    }
    std::cout << std::endl;
    std::cout << "sigma b map:"<<std::endl;
    for(const auto& val : sigma_map_b){
        std::cout << val.first<<": {"<<val.second[0]<<','<<val.second[1]<<"} ";
    }
    std::cout << std::endl;

    std::map<double,
        std::tuple<int64_t, int64_t, int64_t>
        > radi_map = construct_boundary_radi_map(sigma_map_a, sigma_map_b);
    std::cout << "radi map:"<<std::endl;
    for(const auto& val : radi_map){
        auto [km1_size, k_size, kp1_size] = val.second;
        std::cout << val.first<<": {"<<km1_size<<','<<k_size<<','<<kp1_size<<"} ";
    }
    std::cout << std::endl;
    std::map<double, int64_t> betti_numbers = nt::tda::getBettiNumbers(Boundary_K, Boundary_Kp1, radi_map);
    std::cout << "betti numbers: ";
    for(const auto& val : betti_numbers){
        std::cout << "{"<<val.first<<","<<val.second<<"} "<<std::endl;
    }
    std::cout << std::endl;
}


void persistent_diagram_test(){
    persistent_pointcloud_2d();
    // unit_simultaneous_test(3);
    // simultaneous_betti_number_test(); 
}


