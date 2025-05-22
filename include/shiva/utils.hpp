#ifndef SHIVA_UTILS_HPP
#define SHIVA_UTILS_HPP

#include <errno.h>
#include <sys/socket.h>
#include <vector>

#include "shiva/exceptions.hpp"

namespace shiva
{
    namespace utils
    {
        /***
         * Check if the machine is big endian or little endian
         * by checking the first byte of an integer.
         *
         * @return true if the machine is big endian, false otherwise.
         */
        inline bool IsBigEndianMachine()
        {
            unsigned int x = 1;
            return !(bool)*((char *)&x);
        }

        /***
         * Toggle the endianness of a value.
         *
         * @param value The value to toggle.
         * @return The value with the endianness toggled.
         */
        template <typename T> inline T ToggleEndianness(const T &value)
        {
            T result;
            char *pValue = (char *)&value;
            char *pResult = (char *)&result;
            int size = sizeof(T);
            for (int i = 0; i < size; i++)
            {
                pResult[i] = pValue[size - 1 - i];
            }

            return result;
        }

        /***
         * Convert a vector of data to big endian, only if the machine is little endian.
         *
         * @param data The data to convert.
         * @return The data in big endian.
         */
        template <typename T>
        inline std::vector<T> ToBigEndian(const std::vector<T> &data)
        {
            if (IsBigEndianMachine())
            {
                return data;
            }

            std::vector<T> result;
            result.reserve(data.size());
            for (auto it = data.begin(); it != data.end(); ++it)
            {
                result.emplace_back(ToggleEndianness(*it));
            }
            return result;
        }

        /***
         * Convert a vector of data from big endian to the machine's endianness.
         *
         * @param data The data to convert.
         * @return The data in the machine's endianness.
         */
        template <typename T>
        inline std::vector<T> FromBigEndian(const std::vector<T> &data)
        {
            if (IsBigEndianMachine())
            {
                return data;
            }

            std::vector<T> result;
            result.reserve(data.size());
            for (auto it = data.begin(); it != data.end(); ++it)
            {
                result.emplace_back(ToggleEndianness(*it));
            }
            return result;
        }

        /***
         * Receive data from a socket.
         *
         * @param sock The socket to receive data from.
         * @param buffer The buffer to store the received data.
         * @param size The size of the data to receive.
         * @param msg_name The name of the message being received.
         */
        inline void SocketRecv(int sock, uint8_t *buffer, int size,
                               std::string msg_name)
        {
            int received_size = 0;
            while (received_size < size)
            {
                int remains = size - received_size;
                int chunk_size = recv(sock, buffer + received_size, remains, 0);
                if (chunk_size <= 0)
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK)
                    {
                        std::string msg = "Timeout while receiving " + msg_name;
                        throw ShivaTimeoutException(msg);
                    }
                    std::string msg = "Error while receiving " + msg_name;
                    throw std::runtime_error(msg);
                }
                received_size += chunk_size;
            }
        }

        /***
         * Send data to a socket.
         *
         * @param sock The socket to send data to.
         * @param buffer The buffer containing the data to send.
         * @param size The size of the data to send.
         * @param msg_name The name of the message being sent.
         */
        inline void SocketSend(int sock, const uint8_t *buffer, int size,
                               std::string msg_name)
        {
            int sent_size = 0;
            while (sent_size < size)
            {
                int remains = size - sent_size;
                int chunk_size = send(sock, buffer + sent_size, remains, 0);
                if (chunk_size <= 0)
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK)
                    {
                        std::string msg = "Timeout while sending " + msg_name;
                        throw ShivaTimeoutException(msg);
                    }
                    std::string msg = "Error while sending " + msg_name;
                    throw std::runtime_error(msg);
                }
                sent_size += chunk_size;
            }
        }
    }
}

#endif // SHIVA_UTILS_HPP
