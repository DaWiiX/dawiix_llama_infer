#include "../../include/base/base.h"
#include <string>

namespace base
{
    Status::Status(int code, std::string err_message)
    :code_(code), message_(err_message)
    {}

    Status& Status::operator=(int code)
    {
        this->code_ = code;
        return *this;
    };

    bool Status::operator==(int code) const
    {
        if (this->code_ == code) return true;
        else return false;
    };

    bool Status::operator!=(int code) const
    {
        if (this->code_ != code) return true;
        else return false;
    };

    Status::operator int() const { return this->code_; };

    Status::operator bool() const { return this->code_ == StatusCode::Success; }

    int32_t Status::get_err_code() const { return this->code_; }

    const std::string& Status::get_err_msg() const { return message_; }

    void Status::set_err_msg(const std::string& err_msg)
    {
        this->message_ = err_msg;
        return;
    }

    namespace error
    {
        Status Success(const std::string& err_msg)
        {
            return Status{StatusCode::Success, err_msg};
        }

        Status FunctionNotImplement(const std::string& err_msg)
        {
            return Status{StatusCode::FunctionNotImplement, err_msg};
        }

        Status PathNotValid(const std::string& err_msg)
        {
            return Status{StatusCode::PathNotValid, err_msg};
        }

        Status ModelParseError(const std::string& err_msg)
        {
            return Status{StatusCode::ModelParseError, err_msg};
        }

        Status InternalError(const std::string& err_msg)
        {
            return Status{StatusCode::InternalError, err_msg};
        }

        Status KeyValueHasExist(const std::string& err_msg)
        {
            return Status{StatusCode::KeyValueHasExist, err_msg};
        }

        Status InvalidArgument(const std::string& err_msg)
        {
            return Status{StatusCode::InvalidArgument, err_msg};
        }

        std::ostream& operator<<(std::ostream& os, const Status& x)
        {
            os << "Status_code:" << x.get_err_code() << ' ' << "Error_msg:" << x.get_err_msg();
            return os;
        }
    }
}