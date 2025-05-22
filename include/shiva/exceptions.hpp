#include <cstring>
#include <exception>
#include <iostream>

namespace shiva
{
    class ShivaTimeoutException : public std::exception
    {
    private:
        std::string m_message;

    public:
        ShivaTimeoutException(const std::string &msg) : m_message(msg) {}

        const char *what() const noexcept override { return m_message.c_str(); }
    };
}
