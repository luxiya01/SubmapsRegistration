#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "feature_matching/cxxopts.hpp"
#include "feature_matching/utils_visualization.hpp"
#include "feature_matching/corresp_matching.hpp"

#include "yaml-cpp/yaml.h"
#include "yaml-cpp/parser.h"

using namespace std;

PointCloudT::Ptr load_point_cloud(const string path) {
    boost::filesystem::path cloud_path(path);

    PointCloudT::Ptr cloud_ptr(new PointCloudT);

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

    // Visualization
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->registerKeyboardCallback(&keyboardEventOccurred, (void*) NULL);
    viewer->setBackgroundColor(0.0, 0.0, 0.0);

    int current_viz_step = 0;
    while (!viewer->wasStopped()) {
        viewer->spinOnce();
        if (next_viz_step) {
            cout << "Advance to next viz step..." << endl;
            switch (current_viz_step) {
                case VizStep::init:
                    cout << "Showing initial point clouds..." << endl;
                    break;
                case VizStep::downsampling:
                    cout << "Downsampling point clouds..." << endl;
                    downsample_point_cloud(pc1, yaml_config);
                    downsample_point_cloud(pc2, yaml_config);
                    break;
                case VizStep::gicp:
                    cout << "GICP registration..." << endl;
                    runGicp(pc1, pc2, yaml_config);
                    break;
                default:
                    break;
            }
            rgbVis_two_point_clouds(viewer, pc1, pc2);
            next_viz_step = false;
            current_viz_step += 1;
        }
    }
}
