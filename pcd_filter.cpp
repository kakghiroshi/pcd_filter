//
// Created by usl on 4/6/19.
//
#include <algorithm>
#include <random>
#include <chrono>
#include <ctime>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <fstream>
#include <experimental/filesystem>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include "ceres/ceres.h"
#include <Eigen/Dense>
#include "ceres/rotation.h"
#include "ceres/covariance.h"
#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <termios.h>
#include <unistd.h>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;
typedef pcl::PointXYZ PointT;
double x_min; 
double x_max;
double y_min;
double y_max;
double z_min;
double z_max;
double ransac_threshold;
cv::Mat R;
std::vector<Eigen::Vector3d> lidar_points;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
/*
void cloudHandler(pcl::PointCloud<pcl::PointXYZ>::Ptr& in_cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_filtered) {


    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (in_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;   //欧式聚类对象
    ec.setClusterTolerance (0.12);                     // 设置近邻搜索的搜索半径为0.1m
    ec.setMinClusterSize (100);                 //设置一个聚类需要的最少的点数目为100
    ec.setMaxClusterSize (25000);               //设置一个聚类需要的最大点数目为25000
    ec.setSearchMethod (tree);                    //设置点云的搜索机制
    ec.setInputCloud (in_cloud);
    ec.extract (cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中


    std::vector<int> k_indices;
    std::vector<float> k_sqr_distances;
    tree->nearestKSearch(m_click_point, 1, k_indices, k_sqr_distances);

    visual_chessboard.reset();
    for(unsigned int i = 0; i < cluster_indices.size(); i++)
    {
        int counter = std::count(cluster_indices.at(i).indices.begin(), cluster_indices.at(i).indices.end(), k_indices.at(0));

        if(counter > 0)
        {
            visual_chessboard.m_plane_index = i;
            break;
        }
    }

    visual_chessboard.add_color_cloud(m_cloud_ROI, Eigen::Vector3i(255, 255, 255), "cloud_ROI");

    myPointCloud::Ptr temp_cloud (new myPointCloud);
    unsigned int plane_index = 0;
    while(visual_chessboard.m_confirm_flag == false) {

        if (visual_chessboard.update_flag) {

            visual_chessboard.update_flag = false;
            if(visual_chessboard.m_plane_index >=0 && visual_chessboard.m_plane_index < cluster_indices.size())
            plane_index = visual_chessboard.m_plane_index;

            pcl::copyPointCloud<myPoint> (*m_cloud_ROI, cluster_indices.at(plane_index).indices, *temp_cloud);    /// 取出所有的内点
            temp_cloud = getPlane(temp_cloud);    /// 取出平面
            visual_chessboard.add_color_cloud(temp_cloud, Eigen::Vector3i(238, 0, 255), "chess_plane");

            std::cout << "o: confirm;  r: change /click_point; w: plane_idx++; s: plane_idx--;  now plane_inx = " << plane_index << std::endl;
            }
        visual_chessboard.viewer->spinOnce(100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
        visual_chessboard.viewer->removeAllPointClouds();

        if(visual_chessboard.m_reject_flag){
            std::cout << "change /click_point ..." << std::endl;
            return false;
        }

        m_cloud_chessboard = temp_cloud;
        std::cout << "chessboard plane size: " << m_cloud_chessboard->size() << std::endl;
        return true;
}
        
*/


void pointPickingEventOccurred (const pcl::visualization::PointPickingEvent& event, void* viewer_void)
{
    int index = event.getPointIndex ();
    if (index == -1)
    {
        return;
    }
    pcl::PointXYZ clicked_point;
    event.getPoint(clicked_point.x, clicked_point.y, clicked_point.z);
    //float x, y, z;
    //event.getPoint(x, y, z);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(in_cloud);
    pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*>(viewer_void);
    viewer->removeShape("sphere");
    viewer->addSphere(pcl::PointXYZ(clicked_point.x,clicked_point.y, clicked_point.z), 0.01, "sphere", 0);
    viewer->spinOnce();
    std::cout << "Point index: " << index << std::endl;
    std::cout << "Point coordinates: (" << clicked_point.x << ", " << clicked_point.y << ", " << clicked_point.z << ")" << std::endl;

    int K = 10;  // 查询的最近邻数
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    if (tree->nearestKSearch(clicked_point, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
    {
        std::cout << "The closest " << K << " points to the clicked point are: " << std::endl;

        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
            std::cout << "    " << in_cloud->points[pointIdxNKNSearch[i]].x
                      << " " << in_cloud->points[pointIdxNKNSearch[i]].y
                      << " " << in_cloud->points[pointIdxNKNSearch[i]].z
                      << " (distance: " << sqrt(pointNKNSquaredDistance[i]) << ")" << std::endl;
    }
}

int main() {
//读取文件
    cv::FileStorage fs;
    std::string config_yaml = "../config/config.yaml";
    fs.open(config_yaml, cv::FileStorage::READ);
    if ( !fs.isOpened() )
    {
      std::cerr << "can not open " << config_yaml << std::endl;
      return false;
    }
//读取参数
    std::string pcd_folder;
    fs["pcd_path"] >> pcd_folder;
    std::string chessboard_folder;
    fs["chessboard_path"] >> chessboard_folder;
    DIR *dir1 = opendir(pcd_folder.c_str());

    fs["x_min"] >> x_min;
    fs["x_max"] >> x_max;
    fs["y_min"] >> y_min;
    fs["y_max"] >> y_max;
    fs["z_min"] >> z_min;
    fs["z_max"] >> z_max;
    fs["ransac_threshold"] >> ransac_threshold;
    

    //对文件进行遍历
    dirent *entry1;
    std::vector<std::string> files1;
    while ((entry1 = readdir(dir1)) != nullptr) {
        if (entry1->d_type == DT_REG) {
            files1.push_back(entry1->d_name);
        }
    }
//文件夹不存在则创建
    
    if (!fs::is_directory(chessboard_folder)) {
        fs::create_directory(chessboard_folder);
    }
    
    //确定文件夹文件数量
    int file_count = 0;
    for (const auto& entry : fs::directory_iterator(fs::path(pcd_folder))) {
        if (fs::is_regular_file(entry)) {
            ++file_count;
        }
    }
//对点云进行ransac
    /*
    for(int j=0;j<file_count;j++){
        std::string pcd_path = pcd_folder  + files1[j].c_str();
        std::string output_path = chessboard_folder  + files1[j].c_str();
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile(pcd_path, *input_cloud);
        pcl::PointCloud<PointT>::Ptr plane_cloud(new pcl::PointCloud<PointT>);
        cloudHandler(input_cloud,plane_cloud);
        pcl::io::savePCDFileASCII(output_path, *plane_cloud);//保存点云
     
    }
    */
   for(int j=0;j<file_count;j++){
        std::string pcd_path = pcd_folder  + files1[j].c_str();
        std::string output_path = chessboard_folder  + files1[j].c_str();
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile(pcd_path, *input_cloud);
         pcl::visualization::PCLVisualizer viewer("PCL Viewer");
        viewer.addPointCloud(input_cloud, "cloud");

        viewer.registerPointPickingCallback (pointPickingEventOccurred, (void*)&viewer);

        while (!viewer.wasStopped ())
        {
            viewer.spinOnce ();
            
        }
        //pcl::PointXYZ point = input_cloud->points[index];

        
     
    }
    return 0;
}

