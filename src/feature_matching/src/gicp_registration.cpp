#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "feature_matching/cxxopts.hpp"
#include "feature_matching/utils_visualization.hpp"
#include "feature_matching/corresp_matching.hpp"
#include "feature_matching/submaps.hpp"

#include <pcl/point_cloud.h>

#include "data_tools/benchmark.h"

#include "yaml-cpp/yaml.h"
#include "yaml-cpp/parser.h"

using namespace std;

PointCloudT::Ptr load_point_cloud(const string path) {
    boost::filesystem::path cloud_path(path);

    PointCloudT::Ptr cloud_ptr(new PointCloudT);
    cout << "Loading point cloud " << cloud_path.string() << endl;

    if (pcl::io::loadPCDFile(cloud_path.string(), *cloud_ptr) < 0) {
        PCL_ERROR("Error loading cloud %s.\n", cloud_path.string());
    }
    return cloud_ptr;
}

void rgbVis_two_point_clouds(pcl::visualization::PCLVisualizer::Ptr& viewer,
    PointCloudT::Ptr& pc1, PointCloudT::Ptr& pc2) {
    viewer->removeAllPointClouds();
    rgbVis(viewer, pc1, 0);
    rgbVis(viewer, pc2, 1);
}

PointsT point_clouds_to_pointsT(const vector<PointCloudT::Ptr> point_clouds) {
    PointsT pc_in_pointsT;
    for (const PointCloudT::Ptr& pc : point_clouds) {
        Eigen::MatrixXf pc_matrix = pc->getMatrixXfMap(3,4,0).transpose();
        pc_in_pointsT.push_back(pc_matrix.cast<double>());
    }
    return pc_in_pointsT;
}

int main(int argc, char **argv) {
    std::string submap1, submap2, config;
    cxxopts::Options options("GICP registration", "GICP registration for 2 submaps");
    options.add_options()("help", "Print help")
    ("submap1", "PCD map 1", cxxopts::value(submap1))
    ("submap2", "PCD map 2", cxxopts::value(submap2))
    ("config", "YAML config file for GICP parameters", cxxopts::value(config));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    // Load the yaml file
    boost::filesystem::path yaml_path(config);
    YAML::Node yaml_config = YAML::LoadFile(yaml_path.string());

    PointCloudT::Ptr pc1 = load_point_cloud(submap1);
    PointCloudT::Ptr pc2 = load_point_cloud(submap2);

    // Add benchmarks
    cout << "Point clouds vec construction..." << endl;
    vector<PointCloudT::Ptr> point_clouds_vec{pc1, pc2};
    cout << "Create benchmark..." << endl;
    benchmark::track_error_benchmark benchmark = benchmark::track_error_benchmark("Test", yaml_config["benchmark_nbr_rows"].as<int>(),
                                                                                  yaml_config["benchmark_nbr_cols"].as<int>());
    cout << "PC to pointsT..." << endl;
    PointsT pc_as_pointsT = point_clouds_to_pointsT(point_clouds_vec);
    cout << "benchmark.track_img_params..." << endl;
    benchmark.track_img_params(pc_as_pointsT);
    cout << "After benchmark.track_img_params..." << endl;

    // Visualization
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->registerKeyboardCallback(&keyboardEventOccurred, (void*) NULL);
    viewer->setBackgroundColor(0.0, 0.0, 0.0);

    int current_viz_step = 0;
    int gicp_current_iteration = 0;
    int gicp_max_iterations = yaml_config["gicp_max_iterations"].as<int>();
    Matrix4f final_transform = Matrix4f::Identity();
    float min_consistency_error = std::numeric_limits<float>::max();

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
        if (next_viz_step) {
            cout << "Advance to next viz step..." << endl;
            switch (current_viz_step) {
                case VizStep::init:
                    cout << "Showing initial point clouds..." << endl;
                    break;
                case VizStep::gicp:
                    cout << "GICP registration..." << endl;
                    while (gicp_current_iteration < gicp_max_iterations) {
                        Matrix4f transform = runGicp(pc1, pc2, yaml_config);
                        string benchmark_name = "gicp_" + std::to_string(gicp_current_iteration);
                        // Update benchmark: note that the second param in benchmark.add_benchmark is not actually used
                        pc_as_pointsT = point_clouds_to_pointsT(point_clouds_vec);
                        benchmark.add_benchmark(pc_as_pointsT, pc_as_pointsT, benchmark_name);

                        if (transform == Matrix4f::Identity()) {
                            break;
                        } else if (benchmark.consistency_rms_errors[benchmark_name] > min_consistency_error) {
                            // Transform pc1 back to before the current GICP transform was performed
                            pcl::transformPointCloud(*pc1, *pc1, transform.inverse());
                            break;
                        }
                        min_consistency_error = benchmark.consistency_rms_errors[benchmark_name];
                        final_transform*=transform;
                        gicp_current_iteration++;
                        cout << "Final transform: \n" << final_transform << endl;
                    }
                    break;
                default:
                    break;
            }
            // Update visualization
            rgbVis_two_point_clouds(viewer, pc1, pc2);
            // Update benchmark: note that the second param in benchmark.add_benchmark is not actually used
            pc_as_pointsT = point_clouds_to_pointsT(point_clouds_vec);
            string benchmark_name = viz_step_to_string[current_viz_step];
            benchmark.add_benchmark(pc_as_pointsT, pc_as_pointsT, benchmark_name);

            next_viz_step = false;
            current_viz_step += 1;
            if (current_viz_step > VizStep::gicp) {
                return 0;
            }
        }
    }
}
