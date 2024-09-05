#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../include/base/base.h"

// 测试默认构造函数
TEST(StatusTest, DefaultConstructor) {
    base::Status status;
    EXPECT_EQ(status.get_err_code(), base::StatusCode::Success);  // 检查是否为成功状态
    EXPECT_EQ(status.get_err_msg(), "");  // 检查默认错误信息是否为空
}

// 测试带参数的构造函数
TEST(StatusTest, ParameterizedConstructor) {
    base::Status status(base::StatusCode::InvalidArgument, "Invalid argument");
    EXPECT_EQ(status.get_err_code(), base::StatusCode::InvalidArgument);  // 检查错误码
    EXPECT_EQ(status.get_err_msg(), "Invalid argument");  // 检查错误信息
}

// 测试赋值操作符
TEST(StatusTest, AssignmentOperator) {
    base::Status status;
    status = base::StatusCode::PathNotValid;
    EXPECT_EQ(status.get_err_code(), base::StatusCode::PathNotValid);
}

// 测试相等和不等操作符
TEST(StatusTest, EqualityOperator) {
    base::Status status(base::StatusCode::Success);
    EXPECT_TRUE(status == base::StatusCode::Success);  // 相等
    EXPECT_FALSE(status != base::StatusCode::Success);  // 不等
}

// 测试设置错误消息
TEST(StatusTest, SetErrorMessage) {
    base::Status status;
    status.set_err_msg("An error occurred");
    EXPECT_EQ(status.get_err_msg(), "An error occurred");
}