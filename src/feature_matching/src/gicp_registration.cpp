#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>

#include "feature_matching/cxxopts.hpp"
#include "feature_matching/utils_visualization.hpp"
#include "feature_matching/corresp_matching.hpp"
#include "feature_matching/submaps.hpp"

#include <pcl/point_cloud.h>

#include "data_tools/benchmark.h"

#include "yaml-cpp/yaml.h"
#include "yaml-cpp/parser.h"

using namespace std;

PointCloudT::Ptr load_point_cloud(const boost::filesystem::path cloud_path) {
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

double add_benchmark_and_update_min_vals(benchmark::track_error_benchmark& benchmark,
                                       const string& current_benchmark_name,
                                       string& min_benchmark_name,
                                       double min_consistency_error,
                                       const vector<PointCloudT::Ptr>& point_clouds_vec){
    // Update benchmark: note that the second param in benchmark.add_benchmark is not actually used
    PointsT pc_as_pointsT = point_clouds_to_pointsT(point_clouds_vec);
    benchmark.add_benchmark(pc_as_pointsT, pc_as_pointsT, current_benchmark_name);

    if (benchmark.consistency_rms_errors[current_benchmark_name] < min_consistency_error) {
        min_consistency_error = benchmark.consistency_rms_errors[current_benchmark_name];
        min_benchmark_name = current_benchmark_name;
    }
    printf("Min benchmark name: %s, min RMS: %.5f\n", min_benchmark_name.c_str(), min_consistency_error);
    return min_consistency_error;
}

int main(int argc, char **argv) {
    std::string submap1_str, submap2_str, config;
    cxxopts::Options options("GICP registration", "GICP registration for 2 submaps");
    options.add_options()("help", "Print help")
    ("submap1", "PCD map 1", cxxopts::value(submap1_str))
    ("submap2", "PCD map 2", cxxopts::value(submap2_str))
    ("config", "YAML config file for GICP parameters", cxxopts::value(config));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    // Load the yaml file
    boost::filesystem::path submap1(submap1_str);
    boost::filesystem::path submap2(submap2_str);
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
    bool auto_viz = yaml_config["auto_viz"].as<bool>();
    next_viz_step = auto_viz;

    int current_viz_step = 0;
    int gicp_current_iteration = 0;
    int gicp_max_iterations = yaml_config["gicp_max_iterations"].as<int>();
    Matrix4f final_transform = Matrix4f::Identity();
    double min_consistency_error = std::numeric_limits<double>::max();
    string current_benchmark_name = "";
    string min_benchmark_name = "";
    string submap_pair = submap1.stem().string() + "-" + submap2.stem().string();
    ofstream out_file(submap_pair + ".yaml");

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
        if (next_viz_step) {
            cout << "Advance to next viz step..." << endl;
            switch (current_viz_step) {
                case VizStep::init:
                    cout << "Showing initial point clouds..." << endl;
                    current_benchmark_name = submap_pair + "_" + viz_step_to_string[current_viz_step];
                    min_consistency_error = add_benchmark_and_update_min_vals(benchmark, current_benchmark_name, min_benchmark_name,
                                                      min_consistency_error, point_clouds_vec);
                    break;
                case VizStep::gicp:
                    cout << "GICP registration..." << endl;
                    while (gicp_current_iteration < gicp_max_iterations) {
                        Matrix4f transform = runGicp(pc1, pc2, yaml_config);
                        if (transform == Matrix4f::Identity()) {
                            break;
                        }
                        current_benchmark_name = submap_pair + "_gicp_" + std::to_string(gicp_current_iteration);
                        min_consistency_error = add_benchmark_and_update_min_vals(benchmark, current_benchmark_name, min_benchmark_name,
                                                        min_consistency_error, point_clouds_vec);
                        if (benchmark.consistency_rms_errors[current_benchmark_name] > min_consistency_error) {
                            // Transform pc1 back to before the current GICP transform was performed
                            cout << "\n\nTransforming back!!!\n"
                                 << "Min_consistency_error: " << min_consistency_error << "\n"
                                 << "current error: " << benchmark.consistency_rms_errors[current_benchmark_name]
                                 << "\n\n"
                                 << endl;

                            pcl::transformPointCloud(*pc1, *pc1, transform.inverse());
                            break;
                        }
                        final_transform*=transform;
                        gicp_current_iteration++;
                        cout << "Final transform: \n" << final_transform << endl;
                    }

                    // Write GICP results to file
                    if (out_file.is_open()) {
                        out_file << "no_gicp_iterations: " << gicp_current_iteration << endl;
                        out_file << "rms_error:\n" 
                                 << "  init: " << benchmark.consistency_rms_errors[submap_pair+"_init"] << endl 
                                 << "  final: " <<benchmark.consistency_rms_errors[min_benchmark_name] << endl;
                        out_file << "std_all:\n"
                                 << "  init: " << benchmark.std_grids_with_hits[submap_pair+"_init"] << endl
                                 << "  final: " << benchmark.std_grids_with_hits[min_benchmark_name] << endl;
                        out_file << "std_overlap:\n"
                                 << "  init: " << benchmark.std_grids_with_overlaps[submap_pair+"_init"] << endl
                                 << "  final: " << benchmark.std_grids_with_overlaps[min_benchmark_name] << endl;
                        out_file << "transform: [";
                        int num = 0;
                        for (float tf : final_transform.reshaped<RowMajor>()) {
                            out_file << tf;
                            if (num < final_transform.size()-1) {
                                out_file << ", ";
                            } else {
                                out_file << "]" << endl;
                            }
                            num ++;
                        }
                    }
                default:
                    break;
            }
            // Update visualization
            rgbVis_two_point_clouds(viewer, pc1, pc2);
            next_viz_step = auto_viz;
            current_viz_step += 1;
            if (current_viz_step > VizStep::gicp) {
                return 0;
            }
        }
    }
}
