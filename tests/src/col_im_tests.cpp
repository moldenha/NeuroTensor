#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"


// There are no number changes really just placement
// for that reason multiply dtype are not tested

void col_im_test(){
    using namespace nt::literals;
    // unfold1d test
    run_test("unfold1d", [] {
        nt::Tensor x = nt::randn({2, 3, 10}); // [N, C, L]
        auto y = nt::unfold1d(x, kernel_size = 3, padding = 1, stride = 2, dilation = 2);

        nt::utils::throw_exception(!x.is_null(), "x is null after unfold1d");
        nt::utils::throw_exception(y.shape().size() == 3, "unfold1d should return a 3D tensor");

        int64_t expected_L = (10 + 2 * 1 - 2 * (3 - 1) - 1) / 2 + 1; // output length
        nt::utils::throw_exception(y.shape() == nt::SizeRef({2, 3 * 3, expected_L}), "unfold1d output shape incorrect");
    });
    // unfold test
    run_test("unfold2d", [] {
        nt::Tensor x = nt::rand(0, 10, {1, 3, 8, 8}); // [N, C, H, W]
        auto y = nt::unfold2d(x, {3, 2}, padding = 1, stride = {2, 2}, dilation = {1, 2});

        int64_t H = 8, W = 8;
        int64_t H_out = (H + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1;
        int64_t W_out = (W + 2 * 1 - 2 * (2 - 1) - 1) / 2 + 1;

        nt::utils::throw_exception(!y.is_null(), "unfold output is null");
        nt::utils::throw_exception(y.shape() == nt::SizeRef({1, 3 * 3 * 2, H_out * W_out}), "unfold2d output shape incorrect");
    });

    run_test("unfold3d", [] {
        nt::Tensor x = nt::rand(0, 10, {1, 2, 6, 6, 6}); // [N, C, D, H, W]
        auto y = nt::unfold3d(x, {3, 3, 3}, padding = {1, 1, 1}, stride = {1, 2, 2}, dilation = 1);

        int64_t D = 6, H = 6, W = 6;
        int64_t D_out = (D + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
        int64_t H_out = (H + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1;
        int64_t W_out = (W + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1;

        nt::utils::throw_exception(y.shape() == nt::SizeRef({1, 2 * 3 * 3 * 3, D_out * H_out * W_out}), "unfold3d shape incorrect");
    });

    run_test("unfoldnd -> unfold1d", []{
        nt::Tensor x = nt::randn({2, 3, 10});
        auto y = nt::unfold1d(x, kernel_size=3, padding=1, stride=2, dilation=2);
        auto y2 = nt::functional::unfoldnd(x, 1, /*kernel_size = */ 3,
                                           /*dilation = */ 2, 
                                           /*padding = */ 1, 
                                           /*stride = */ 2,
                                           /*transpose_out = */ true,
                                           /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y, y2),
                                   "Error, output from unfold1d does not match output from unfoldnd in 1d mode"
                                    "$ \n$ \n$", nt::noprintdtype, y, y2);
    });

    run_test("unfoldnd -> unfold2d", []{
        nt::Tensor x = nt::rand(0, 10, {1, 3, 8, 8}); // [N, C, H, W]
        auto y = nt::unfold2d(x, {3, 2}, padding = 1, stride = {2, 2}, dilation = {1, 2});
        auto y2 = nt::functional::unfoldnd(x, 2, /*kernel_size = */ {3, 2},
                                           /*dilation = */ {1, 2}, 
                                           /*padding = */ 1, 
                                           /*stride = */ {2, 2},
                                           /*transpose_out = */ true,
                                           /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y, y2),
                                   "Error, output from unfold2d does not match output from unfoldnd in 2d mode");
    });
    run_test("unfoldnd -> unfold3d", []{
        nt::Tensor x = nt::rand(0, 10, {1, 2, 6, 6, 6}); // [N, C, D, H, W]
        auto y = nt::unfold3d(x, {3, 3, 3}, padding = {1, 1, 1}, stride = {1, 2, 2}, dilation = 1);
        auto y2 = nt::functional::unfoldnd(x, 3, /*kernel_size = */ {3, 3, 3},
                                           /*dilation = */ 1, 
                                           /*padding = */ {1, 1, 1}, 
                                           /*stride = */ {1, 2, 2},
                                           /*transpose_out = */ true,
                                           /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y, y2),
                                   "Error, output from unfold3d does not match output from unfoldnd in 3d mode");
    });

    // fold test
    run_test("fold2d", [] {
        nt::Tensor x = nt::randn({3, 18, 25});
        auto y = nt::fold2d(x, kernel_size = {3, 2}, output_size = 5, padding = 1, dilation = {1,2});
    });
    run_test("foldnd -> fold2d", []{
        nt::Tensor x = nt::randn({3, 18, 25});
        auto y = nt::fold2d(x, kernel_size = {3, 2}, output_size = 5, padding = 1, dilation = {1,2});
        auto y2 = nt::functional::foldnd(x, 2, /*output_size = */ 5, /*kernel_size = */{3, 2}, /*dilation = */ {1, 2}, /*padding = */1,
                                         /*stride = */1, /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y, y2),
                                   "Error, output from fold2d does not match output from unfoldnd in 2d mode");

    });

    run_test("fold1d -> foldnd", []{
        nt::Tensor x = nt::randn({3, 18, 3});
        auto y = nt::fold1d(x, kernel_size = 3, output_size = 5, padding = 1, dilation = 2);
        auto y2 = nt::functional::foldnd(x, 1, /*output_size = */ 5, /*kernel_size = */3, /*dilation = */ 2, /*padding = */1,
                                         /*stride = */1, /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y, y2),
                                   "Error, output from fold1d does not match output from unfoldnd in 1d mode");

    });

    run_test("fold3d -> foldnd", []{
        nt::Tensor x = nt::randn({3, 24, 150});
        auto y = nt::fold3d(x, kernel_size = {3, 2, 2}, output_size = 5, padding = 1, dilation = {1, 1, 2});
        auto y2 = nt::functional::foldnd(x, 3, /*output_size = */ 5, /*kernel_size = */{3, 2, 2}, /*dilation = */ {1, 1, 2}, /*padding = */1,
                                         /*stride = */1, /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y, y2),
                                   "Error, output from fold3d does not match output from unfoldnd in 3d mode");

    });


}



// int main() {
//     conv_tests();
//     return 0;
// }
