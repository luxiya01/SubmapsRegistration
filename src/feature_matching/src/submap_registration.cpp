#include <feature_matching/corresp_matching.hpp>
#include <feature_matching/utils_visualization.hpp>


void extractKeypointsCorrespondences(const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_2,
                                     CorrespondencesPtr good_correspondences,YAML::Node config)
{
  // Basic correspondence estimation between keypoints
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;
  CorrespondencesPtr all_correspondences(new Correspondences);
  est.setInputTarget(keypoints_1);
  est.setInputSource(keypoints_2);
  double kps_cor_thres = config["kps_cor_thres"].as<double>();

  est.determineReciprocalCorrespondences(*all_correspondences, kps_cor_thres);
  rejectBadCorrespondences(all_correspondences, keypoints_1, keypoints_2, *good_correspondences);

  std::cout << "Number of correspondances " << all_correspondences->size() << std::endl;
  std::cout << "Number of good correspondances " << good_correspondences->size() << std::endl;
}

void extractFeaturesCorrespondences(const PointCloud<SHOT352>::Ptr &shot_src,
                                    const PointCloud<SHOT352>::Ptr &shot_trg,
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1,
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_2,
                                    CorrespondencesPtr all_correspondences, YAML::Node config)
{
  // Basic correspondence estimation between keypoints
  pcl::registration::CorrespondenceEstimation<SHOT352, SHOT352> est;
  // CorrespondencesPtr all_correspondences(new Correspondences);
  est.setInputTarget(shot_src);
  est.setInputSource(shot_trg);
  double feats_cor_thres = config["feats_cor_thres"].as<double>();
  
  est.determineReciprocalCorrespondences(*all_correspondences, feats_cor_thres);
  // rejectBadCorrespondences(all_correspondences, keypoints_1, keypoints_2, *good_correspondences);

  std::cout << "Number of correspondances " << all_correspondences->size() << std::endl;
  // std::cout << "Number of good correspondances " << good_correspondences->size() << std::endl;
}

int main(int argc, char **argv)
{
    // Parse the command line arguments for .pcd files
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZ>);
    bool cloud2_given = false;

    // Load the files
    if (pcl::io::loadPCDFile (argv[1], *cloud_1) < 0){
        PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
        return (-1);
    }

    if (argc > 3) {
      if (pcl::io::loadPCDFile (argv[2], *cloud_2) < 0){
          PCL_ERROR ("Error loading cloud %s.\n", argv[2]);
          return (-1);
      }
      cloud2_given = true;
    }

    // Load the yaml file
    YAML::Node config = YAML::LoadFile(argv[argc-1]);
    // cout << "before " << config["harris_kps_radius"] << endl;
    // if (pcl::io::loadPCDFile (argv[2], *cloud_trg) < 0){
    //     PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
    //     return (-1);
    // }

    // Initial noisy misalignment between pointclouds
    std::random_device rd{};
    std::mt19937 seed{rd()};
    
    
    // Construct cloud_2 = cloud_1 + noise if cloud_2 is not given
    if (!cloud2_given) {
      double tf_std_dev = 0.6;
      std::normal_distribution<double> d2{0, tf_std_dev};
      Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
      double theta = M_PI / 10. + d2(seed);
      transformation_matrix (0, 0) = cos (theta);
      transformation_matrix (0, 1) = -sin (theta);
      transformation_matrix (1, 0) = sin (theta);
      transformation_matrix (1, 1) = cos (theta);
      transformation_matrix (0, 3) = -10.;// + d2(seed);
      transformation_matrix (1, 3) = -10.;// + d2(seed);
      transformation_matrix (2, 3) = 0.0;
      pcl::transformPointCloud(*cloud_1, *cloud_2, transformation_matrix);
    }

    // Gaussian noise to points in input clouds
    double pcl_std_dev = 0.01;
    std::normal_distribution<double> d{0, pcl_std_dev};
    for (unsigned int i = 0; i < cloud_1->points.size(); i++)
    {
      cloud_1->points.at(i).x = cloud_1->points.at(i).x + d(seed);
      cloud_1->points.at(i).y = cloud_1->points.at(i).y + d(seed);
      cloud_1->points.at(i).z = cloud_1->points.at(i).z + d(seed);

      cloud_2->points.at(i).x = cloud_2->points.at(i).x + d(seed);
      cloud_2->points.at(i).y = cloud_2->points.at(i).y + d(seed);
      cloud_2->points.at(i).z = cloud_2->points.at(i).z + d(seed);
    }

    // Visualize initial point clouds
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->registerKeyboardCallback(&keyboardEventOccurred, (void*) NULL);
    viewer->setBackgroundColor(0.0, 0.0, 0.0);
    rgbVis(viewer, cloud_1, 0);
    rgbVis(viewer, cloud_2, 1);

    // Placeholder variables
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_2(new pcl::PointCloud<pcl::PointXYZ>);
    // Extract correspondences between keypoints/features
    CorrespondencesPtr good_correspondences(new Correspondences);

    int current_registration_step = 0;
    std::cout << "Submap registration visualization: press space for next step" << std::endl;

    while (!viewer->wasStopped()) {
      viewer->spinOnce();
      if (next_viz_step) {
        next_viz_step = false;
        current_registration_step++;
        if (current_registration_step == 1) {
          std::cout << "Extract keypoints and correspondences between keypoints" << std::endl;
          // Extract keypoints
          harrisKeypoints(cloud_1, *keypoints_1, config);
          harrisKeypoints(cloud_2, *keypoints_2, config);

          // Visualize keypoints
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints1_color_handler(keypoints_1, 255, 0, 0);
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints2_color_handler(keypoints_2, 0, 255, 0);
          viewer->addPointCloud(keypoints_1, keypoints1_color_handler, "keypoints_src", v1);
          viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_src");
          viewer->addPointCloud(keypoints_2, keypoints2_color_handler, "keypoints_trg", v1);
          viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_trg");

          // Compute SHOT descriptors
          PointCloud<SHOT352>::Ptr shot_1(new PointCloud<SHOT352>);
          PointCloud<SHOT352>::Ptr shot_2(new PointCloud<SHOT352>);
          estimateSHOT(keypoints_1, shot_1);
          estimateSHOT(keypoints_2, shot_2);

          // extractKeypointsCorrespondences(keypoints_1, keypoints_2, good_correspondences, config);
          extractFeaturesCorrespondences(shot_1, shot_2, keypoints_1, keypoints_2, good_correspondences, config);

          // Extract correspondences between keypoints
          plotCorrespondences(*viewer, *good_correspondences, keypoints_1, keypoints_2);
        } else if (current_registration_step == 2) {
          std::cout << "Estimate transform based on initial correspondence..." << std::endl;
          // Best transformation between the two sets of keypoints given the remaining correspondences
          Eigen::Matrix4f transform;
          TransformationEstimationSVD<PointXYZ, PointXYZ> trans_est;
          trans_est.estimateRigidTransformation(*keypoints_1, *keypoints_2, *good_correspondences, transform);
          pcl::transformPointCloud(*cloud_2, *cloud_2, transform.inverse());

          viewer->removeAllPointClouds();
          viewer->removeAllShapes();
          rgbVis(viewer, cloud_1, 0);
          rgbVis(viewer, cloud_2, 1);
        } else if (current_registration_step == 3) {
          std::cout << "View GICP registration results" << std::endl;
          // Run GICP
          runGicp(cloud_2, cloud_1);

          viewer->removeAllPointClouds();
          rgbVis(viewer, cloud_1, 0);
          rgbVis(viewer, cloud_2, 1);
        } else {
          std::cout << "No further steps to visualize. Viewing GICP registration results." << std::endl;
        }
      }
    }
    viewer->resetStoppedFlag();
    return 0;
}

