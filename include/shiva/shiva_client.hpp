#ifndef SHIVA_CLIENT_HPP
#define SHIVA_CLIENT_HPP

#include "shiva/shiva_message.hpp"

namespace shiva
{
    class ShivaClient
    {
    public:
        ShivaClient(std::string serverIp, unsigned short serverPort, int timeout_ms = 0)
        {
            this->serverIp = serverIp;
            this->serverPort = serverPort;

            // create TCP socket
            if ((m_sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
                throw std::runtime_error("ShivaClient socket creation failed");

            struct sockaddr_in servAddr;

            // fill server address
            memset(&servAddr, 0, sizeof(servAddr));
            servAddr.sin_family = AF_INET;
            servAddr.sin_addr.s_addr = inet_addr(serverIp.c_str());
            servAddr.sin_port = htons(serverPort);

            // connect to server
            if (connect(m_sock, (struct sockaddr *)&servAddr, sizeof(servAddr)) < 0)
                throw std::runtime_error("ShivaClient connect failed");

            // set TCP_NODELAY
            int enable_no_delay = 1;
            if (setsockopt(m_sock, IPPROTO_TCP, TCP_NODELAY, &enable_no_delay,
                           sizeof(int)) < 0)
                throw std::runtime_error("ShivaClient setsockopt TCP_NODELAY failed");

            if (timeout_ms > 0)
            {
                // set timeout (both send and receive)
                struct timeval tv;
                tv.tv_sec = timeout_ms / 1000;
                tv.tv_usec = (timeout_ms % 1000) * 1000;
                if (setsockopt(m_sock, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv,
                               sizeof(struct timeval)) < 0)
                    throw std::runtime_error(
                        "ShivaClient setsockopt SO_SNDTIMEO failed");
                if (setsockopt(m_sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv,
                               sizeof(struct timeval)) < 0)
                    throw std::runtime_error(
                        "ShivaClient setsockopt SO_RCVTIMEO failed");
            }

            // wait for the handshake
            usleep(10000);
        }

        ShivaMessage sendAndReceiveMessage(ShivaMessage &message)
        {
            message.sendMessage(m_sock);
            return ShivaMessage::receive(m_sock);
        }

        void close() { ::close(m_sock); }

        ~ShivaClient() { close(); }

        std::string serverIp;
        unsigned short serverPort;

    private:
        int m_sock;
    };
}

#endif // SHIVA_CLIENT_HPP
