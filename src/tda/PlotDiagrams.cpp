#include "PlotDiagrams.h"

namespace nt {
namespace tda {

void plotPersistentDiagram(
    const std::vector<std::vector<std::tuple<Tensor, double, double>>>
        &homologyData) {
    using namespace matplot;
    // blue, orange, green, red, magenta
    std::vector<std::array<float, 3>> colors = {{0.12f, 0.46f, 0.7f},
                                                {1.0f, 0.49f, 0.055f},
                                                {0.17f, 0.63f, 0.17f},
                                                {1.0f, 0.0f, 0.0f},
                                                {1.0f, 0.0f, 1.0f}};
    while (colors.size() < homologyData.size()) {
        colors.push_back(
            {static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
             static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
             static_cast<float>(rand()) / static_cast<float>(RAND_MAX)});
    }

    double max_radi = 0;
    for (const auto &group : homologyData) {
        for (const auto &tup : group) {
            max_radi = std::max(max_radi, std::get<2>(tup));
        }
    }

    std::vector<std::string> names;
    figure();
    hold(on); // Ensure multiple scatter plots are overlaid
    std::vector<double> x_vals = {0, max_radi * 1.1};
    std::vector<double> y_vals = {0, max_radi * 1.1};
    std::vector<double> x_vals2 = {0, max_radi * 1.1};
    std::vector<double> y_vals2 = {max_radi, max_radi};
    auto diag_line =
        plot(x_vals, y_vals, "k--"); // "k--" means black dotted line
    diag_line->line_width(2.0);
    names.push_back("∞");
    int64_t counter = 0;
    for (const auto &group : homologyData) {
        std::vector<double> births, deaths;
        for (const auto &tup : group) {
            auto [simplex_complex, birth, death] = tup;
            births.push_back(birth);
            deaths.push_back(death == -1 ? max_radi : death);
        }

        if (births.size() == 0) {
            ++counter;
            continue;
        }
        auto sc = scatter(births, deaths, 10); // Scatter plot
        sc->marker_color(colors[counter]);
        // sc->marker("o");
        sc->marker_face_color(colors[counter]);
        names.push_back("H_{" + std::to_string(counter) + "}");
        // sc->display_name("H" + std::to_string(i));
        ++counter;
    }
    auto straight_line = plot(x_vals2, y_vals2, "k--");
    straight_line->line_width(2.0);

    xlabel("Birth");
    ylabel("Death");
    title("Persistence Diagram");
    axis({-0.5, max_radi * 1.1, -0.5, max_radi * 1.1});
    grid(false);
    auto l = matplot::legend(names); // Show legend
    l->location(legend::general_alignment::bottomright);
    l->font_size(15);

    // save("homology_diagram.png");
    // if(_do_show){show();}
}

void plotBarcode(
    const std::vector<std::vector<std::tuple<Tensor, double, double>>>
        &homologyData) {
    using namespace matplot;

    // Colors for different homology groups
    std::vector<std::array<float, 3>> colors = {
        {0.12f, 0.46f, 0.7f},  // blue
        {1.0f, 0.49f, 0.055f}, // orange
        {0.17f, 0.63f, 0.17f}, // green
        {1.0f, 0.0f, 0.0f},    // red
        {1.0f, 0.0f, 1.0f}     // magenta
    };

    while (colors.size() < homologyData.size()) {
        colors.push_back({static_cast<float>(rand()) / RAND_MAX,
                          static_cast<float>(rand()) / RAND_MAX,
                          static_cast<float>(rand()) / RAND_MAX});
    }

    double max_radi = 0;
    for (const auto &group : homologyData) {
        for (const auto &tup : group) {
            max_radi = std::max(max_radi, std::get<2>(tup));
        }
    }

    figure();
    hold(on);

    std::vector<std::string> names;
    int64_t counter = 0;
    double y_offset = 0.1;
    double bar_height = 0.8;
    std::vector<std::vector<double>> y_offsets(homologyData.size());
    for (const auto &group : homologyData) {
        y_offsets[counter] = std::vector<double>(group.size(), 0);
        for (int i = 0; i < group.size(); ++i) {
            y_offsets[counter][i] = y_offset;
            y_offset += bar_height;
        }
        ++counter;
    }
    counter = 0;
    // doing each homology group first ensures the correct color for the legend
    for (const auto &group : homologyData) {
        if (group.size() == 0) {
            ++counter;
            continue;
        }
        auto [simplex_complex, birth, death] = group[0];
        death = (death == -1) ? max_radi : death;

        std::vector<double> x = {birth, death};
        std::vector<double> y = {y_offsets[counter][0], y_offsets[counter][0]};

        auto line = plot(x, y, "-");
        line->line_width(4.0);
        line->color(colors[counter]);
        names.push_back("H_{" + std::to_string(counter) + "}");
        ++counter;
    }
    counter = 0;
    for (const auto &group : homologyData) {
        for (size_t i = 0; i < group.size(); ++i) {
            auto [simplex_complex, birth, death] = group[i];
            death = (death == -1) ? max_radi : death;

            std::vector<double> x = {birth, death};
            std::vector<double> y = {y_offsets[counter][i],
                                     y_offsets[counter][i]};

            auto line = plot(x, y, "-");
            line->line_width(4.0);
            line->color(colors[counter]);
        }
        ++counter;
    }

    xlabel("Filtration Value");
    // ylabel("Homology Classes");
    title("Barcode for Persistent Homology");

    axis({-0.5, max_radi * 1.1, -0.5, y_offset});
    grid(false);

    auto l = matplot::legend(names);
    l->location(legend::general_alignment::bottomright);
    l->font_size(15);

    // save("homology_barcode.png");
    // if (_do_show) {
    // show();
    // }
}

void plotPointCloud(Tensor cloud, int8_t point, int64_t dims) {
    utils::throw_exception(
        dims == 1 || dims == 2 || dims == 3,
        "Can only work with dimensions 1, 2, or 3 but got $", dims);
    using namespace matplot;
    Tensor where = functional::where(cloud == point);
    figure();
    if (dims == 1) {
        Tensor y =
            where[where.numel() - 1].item<Tensor>().to(DType::Double);

        std::vector<double> y_dots(
            reinterpret_cast<double *>(y.data_ptr()),
            reinterpret_cast<double *>(y.data_ptr_end()));
        std::vector<double> x_dots(y_dots.size(), 1);
        scatter(x_dots, y_dots, 7);
        axis({-0.1, static_cast<double>(cloud.shape()[-1]) * 1.1, -0.1,
              static_cast<double>(cloud.shape()[-1]) * 1.1});
    }
    if (dims == 2) {
        Tensor y =
            where[where.numel() - 1].item<Tensor>().to(DType::Double);
        Tensor x =
            where[where.numel() - 2].item<Tensor>().to(DType::Double);
        std::vector<double> y_dots(
            reinterpret_cast<double *>(y.data_ptr()),
            reinterpret_cast<double *>(y.data_ptr_end()));
        std::vector<double> x_dots(
            reinterpret_cast<double *>(x.data_ptr()),
            reinterpret_cast<double *>(x.data_ptr_end()));
        scatter(x_dots, y_dots, 7);
        axis({-0.1, static_cast<double>(cloud.shape()[-1]) * 1.1, -0.1,
              static_cast<double>(cloud.shape()[-2]) * 1.1});
    }
    if (dims == 3) {
        Tensor y =
            where[where.numel() - 1].item<Tensor>().to(DType::Double);
        Tensor x =
            where[where.numel() - 2].item<Tensor>().to(DType::Double);
        Tensor z =
            where[where.numel() - 3].item<Tensor>().to(DType::Double);
        std::vector<double> y_dots(
            reinterpret_cast<double *>(y.data_ptr()),
            reinterpret_cast<double *>(y.data_ptr_end()));
        std::vector<double> x_dots(
            reinterpret_cast<double *>(x.data_ptr()),
            reinterpret_cast<double *>(x.data_ptr_end()));
        std::vector<double> z_dots(
            reinterpret_cast<double *>(z.data_ptr()),
            reinterpret_cast<double *>(z.data_ptr_end()));
        scatter3(z_dots, x_dots, y_dots);
        // s_axis = {-0.1, static_cast<double>(cloud.shape()[-1]) * 1.1,
        //     -0.1, static_cast<double>(cloud.shape()[-2]) * 1.1,
        //     -0.1, static_cast<double>(cloud.shape()[-3]) * 1.1};
    }

    title("Point Cloud");
    grid(false);
    // save("point_cloud.png");
    // if(_do_show){show();}
}

} // namespace tda
} // namespace nt
