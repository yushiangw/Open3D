#include "Open3D/Open3D.h"
#include "PointCloud.h"

using namespace open3d;
int main(int argc, char** argv) {
    // Read point cloud
    auto pcd_legacy = io::CreatePointCloudFromFile(argv[1]);
    auto pcd = tgeometry::PointCloud::FromLegacyPointCloud(
            *pcd_legacy, Dtype::Float32, Device("CUDA:0"));

    utility::Timer timer;
    timer.Start();
    timer.Stop();

    // Benchmark new pointcloud VoxelDownSample
    Tensor a({2, 3}, Dtype::Float32, Device("CUDA:0"));
    a = a + 1;
    (void)a;
    for (int i = 0; i < 10; ++i) {
        timer.Start();
        auto pcd_down = pcd.VoxelDownSample(0.01);
        timer.Stop();
        utility::LogInfo(
                "[Full downsample (including Div)] : takes {} millisecond",
                timer.GetDuration());
    }
}

// Benchmark legacy pointcloud VoxelDownSample
// auto pcd = tgeometry::PointCloud::FromLegacyPointCloud(
//         *pcd_legacy, Dtype::Float32, Device("CUDA:0"));
// timer.Start();
// pcd_legacy->VoxelDownSample(0.01);
// timer.Stop();
// utility::LogInfo("[TestTPointCloud] Legacy VoxelDownSample time: {}",
//                     timer.GetDuration());

// auto pcd_down_legacy = std::make_shared<geometry::PointCloud>(
//         tgeometry::PointCloud::ToLegacyPointCloud(pcd_down));

// utility::LogInfo("pcd size {}", pcd_legacy->points_.size());
// utility::LogInfo("pcd down size {}",
// pcd_down_legacy->points_.size());
// visualization::DrawGeometries({pcd_down_legacy});
