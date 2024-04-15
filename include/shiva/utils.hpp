#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

namespace utils
{
    /***
     * Check if the machine is big endian or little endian
     * by checking the first byte of an integer.
     *
     * @return true if the machine is big endian, false otherwise.
     */
    bool IsBigEndianMachine()
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
    template <typename T> T ToggleEndianness(const T &value)
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
    template <typename T> std::vector<T> ToBigEndian(const std::vector<T> &data)
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
    template <typename T> std::vector<T> FromBigEndian(const std::vector<T> &data)
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
}

#endif // UTILS_HPP
