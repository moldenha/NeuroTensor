#include "../src/Tensor.h"
#include "../src/tda/Homology.h"
#include "../src/tda/PlotDiagrams.h"


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



void persistent_diagram_test(){
    persistent_pointcloud_2d(); 
}


