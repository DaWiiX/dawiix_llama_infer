#include <glog/logging.h>
#include <gtest/gtest.h>
#include <filesystem>  // C++17

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("DaWiiX");
    std::filesystem::create_directories("./log/");
    FLAGS_log_dir = "./log/";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Starting tests...\n";
    return RUN_ALL_TESTS();
}