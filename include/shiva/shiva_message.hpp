#ifndef SHIVA_MESSAGE_HPP
#define SHIVA_MESSAGE_HPP

#include <arpa/inet.h>
#include <cstdint>
#include <iostream>
#include <netinet/tcp.h>
#include <nlohmann/json.hpp>
#include <string>
#include <typeindex>
#include <unistd.h>
#include <unordered_map>

#include "shiva/utils.hpp"

namespace shiva
{
    /**
     * Bigendian uint32_t
     */
    class be_uint32_t
    {
    public:
        be_uint32_t() : be_val_(0) {}
        be_uint32_t(const uint32_t &val) : be_val_(htonl(val)) {}
        operator uint32_t() const { return ntohl(be_val_); }

    private:
        uint32_t be_val_;
    } __attribute__((packed));

    std::unordered_map<std::type_index, int8_t> TensorTypeMap = {
        {typeid(float), 1},         // 32-bit floating point
        {typeid(uint8_t), 3},       // 8-bit unsigned integer
        {typeid(int8_t), 4},        // 8-bit signed integer
        {typeid(uint16_t), 5},      // 16-bit unsigned integer
        {typeid(int16_t), 6},       // 16-bit signed integer
        {typeid(uint32_t), 7},      // 32-bit unsigned integer
        {typeid(int), 8},           // 32-bit signed integer
        {typeid(unsigned long), 9}, // 64-bit unsigned integer
        {typeid(long), 10},         // 64-bit signed integer
        {typeid(double), 11},       // double
        {typeid(long double), 12},  // long double
        {typeid(long long), 13},    // long long

        /*
        The following entry has been removed since it was duplicated. It was originally
        there following the Python implementation, but it turned out to be an error.
        {typeid(double), 2}, // 64-bit floating point
        */
    };

    struct MessageHeader
    {
        uint8_t MAGIC[4];
        be_uint32_t metadata_size;
        uint8_t n_tensors;
        uint8_t trail_size;
        uint8_t CRC;
        uint8_t CRC2;

        MessageHeader() {}
        MessageHeader(int metadata_size, uint8_t n_tensors, uint8_t trail_size)
        {
            // control code
            this->MAGIC[0] = 6;
            this->MAGIC[1] = 66;
            this->MAGIC[2] = 11;
            this->MAGIC[3] = 1;

            // payload size
            this->metadata_size = metadata_size;
            this->trail_size = trail_size;
            this->n_tensors = n_tensors;

            // compute crc summing all
            this->CRC = 0;
            this->CRC +=
                this->MAGIC[0] + this->MAGIC[1] + this->MAGIC[2] + this->MAGIC[3];
            this->CRC += this->metadata_size;
            this->CRC += this->n_tensors;
            this->CRC += this->trail_size;
            this->CRC = this->CRC % 256;

            // compute crc2 summing all previous
            this->CRC2 = (this->CRC + this->CRC) % 256;
        }
    };

    struct TensorHeader
    {
        uint8_t rank = 0;
        uint8_t dtype = 0;
    };

    class BaseTensor
    {
    public:
        std::vector<uint32_t> shape;
        std::type_index type;
        TensorHeader header;

        BaseTensor() : type(typeid(float)) {}
        virtual ~BaseTensor() = default;

        TensorHeader buildHeader()
        {
            TensorHeader header;
            header.rank = this->shape.size();
            header.dtype = TensorTypeMap[type];
            return header;
        }

        void sendHeader(int sock)
        {
            TensorHeader header = this->buildHeader();
            ssize_t size = (ssize_t)sizeof(TensorHeader);
            if (send(sock, &header, size, 0) != size)
                throw std::runtime_error("BaseTensor sendHeader failure");
        }

        void sendShape(int sock)
        {
            if (this->shape.size() == 0)
                return;

            std::vector<be_uint32_t> beshape =
                std::vector<be_uint32_t>(this->shape.begin(), this->shape.end());

            ssize_t size = (ssize_t)sizeof(uint32_t) * beshape.size();
            if (send(sock, &beshape[0], size, 0) != size)
                throw std::runtime_error("BaseTensor sendShape failure");
        }

        virtual void sendData(int sock) = 0;
        virtual void receiveData(int sock) = 0;
    };
    typedef std::shared_ptr<BaseTensor> BaseTensorPtr;

    template <typename T> class Tensor : public BaseTensor
    {
    public:
        std::vector<T> data;
        typedef std::shared_ptr<Tensor<T>> Ptr;

        Tensor() : BaseTensor() { this->type = typeid(T); }
        ~Tensor() {}

        void sendData(int sock)
        {
            if (this->data.size() == 0)
                return;

            std::vector<T> beData = shiva::utils::ToBigEndian(this->data);

            ssize_t size = (ssize_t)sizeof(T) * beData.size();
            if (send(sock, &beData[0], size, 0) != size)
                throw std::runtime_error("Tensor sendData failure");
        }

        void receiveData(int sock)
        {
            if (this->shape.size() == 0)
                return;

            int elements = 1;
            // expected size is product of all shape elements * sizeof(T)
            for (size_t i = 0; i < this->shape.size(); i++)
            {
                elements *= this->shape[i];
            }
            int expected_size = elements * sizeof(T);

            std::shared_ptr<T> response_array(new T[expected_size],
                                              std::default_delete<T[]>());

            T *received_data = response_array.get();
            int received_size = 0;
            while (received_size < expected_size)
            {
                int remains = expected_size - received_size;
                int chunk_size =
                    recv(sock, &(received_data)[received_size], remains, 0);
                received_size += chunk_size;
            }

            std::vector<T> beData = std::vector<T>(elements);
            std::copy_n(received_data, elements, beData.begin());

            this->data.clear();
            this->data = shiva::utils::FromBigEndian(beData);
        }
    };

    class ShivaMessage
    {
    public:
        MessageHeader buildHeader()
        {
            MessageHeader header(this->metadata.dump().size(), this->tensors.size(),
                                 this->namespace_.size());
            return header;
        }

        ShivaMessage() : metadata(nlohmann::json::object()), namespace_(""), tensors()
        {
        }
        ShivaMessage(const ShivaMessage &other)
        {
            this->metadata = other.metadata;
            this->namespace_ = other.namespace_;
            this->tensors = other.tensors;
        }
        ShivaMessage &operator=(const ShivaMessage &other)
        {
            this->metadata = other.metadata;
            this->namespace_ = other.namespace_;
            this->tensors = other.tensors;
            return *this;
        }

        static ShivaMessage receive(int sock)
        {
            ShivaMessage returnMessage;
            MessageHeader returnHeader = returnMessage.receiveHeader(sock);

            for (int i = 0; i < returnHeader.n_tensors; i++)
            {
                TensorHeader th = returnMessage.receiveTensorHeader(sock);
                std::vector<uint32_t> shape =
                    returnMessage.receiveTensorShape(sock, th);

                BaseTensorPtr tensor = returnMessage.receiveTensor(sock, th, shape);
                tensor->header = th;
                tensor->shape = shape;
                returnMessage.tensors.push_back(tensor);
            }
            returnMessage.receiveMetadata(sock, returnHeader.metadata_size);
            returnMessage.receiveNamespace(sock, returnHeader.trail_size);
            return returnMessage;
        }

        void sendMessage(int sock)
        {
            this->sendHeader(sock);
            for (size_t i = 0; i < this->tensors.size(); i++)
            {
                this->tensors[i]->sendHeader(sock);
                this->tensors[i]->sendShape(sock);
                this->tensors[i]->sendData(sock);
            }
            this->sendMetadata(sock);
            this->sendNamespace(sock);
        }

        nlohmann::json metadata;
        std::string namespace_;
        std::vector<std::shared_ptr<BaseTensor>> tensors;

    private:
        MessageHeader receiveHeader(int sock)
        {
            MessageHeader header;
            ssize_t size = (ssize_t)sizeof(MessageHeader);
            if (recv(sock, &header, size, 0) != size)
                throw std::runtime_error("ShivaMessage receiveHeader failure");
            return header;
        }

        TensorHeader receiveTensorHeader(int sock)
        {
            TensorHeader header;
            ssize_t size = (ssize_t)sizeof(TensorHeader);
            if (recv(sock, &header, size, 0) != size)
                throw std::runtime_error("ShivaMessage receiveTensorHeader failure");
            return header;
        }

        std::vector<uint32_t> receiveTensorShape(int sock, TensorHeader &th)
        {

            std::vector<be_uint32_t> beshape(th.rank);
            ssize_t size = (ssize_t)sizeof(be_uint32_t) * th.rank;
            if (recv(sock, &beshape[0], size, 0) != size)
                throw std::runtime_error("ShivaMessage receiveTensorShape failure");
            std::vector<uint32_t> shape(beshape.begin(), beshape.end());
            return shape;
        }

        BaseTensorPtr receiveTensor(int sock, const TensorHeader &th,
                                    const std::vector<uint32_t> &shape)
        {
            BaseTensorPtr tensor;

            switch (th.dtype)
            {
            case 1:
                tensor = std::make_shared<Tensor<float>>();
                break;
            case 2: // this is here only for backwards compatibility, it is not used
                tensor = std::make_shared<Tensor<double>>();
                break;
            case 3:
                tensor = std::make_shared<Tensor<uint8_t>>();
                break;
            case 4:
                tensor = std::make_shared<Tensor<int8_t>>();
                break;
            case 5:
                tensor = std::make_shared<Tensor<uint16_t>>();
                break;
            case 6:
                tensor = std::make_shared<Tensor<int16_t>>();
                break;
            case 7:
                tensor = std::make_shared<Tensor<uint32_t>>();
                break;
            case 8:
                tensor = std::make_shared<Tensor<int>>();
                break;
            case 9:
                tensor = std::make_shared<Tensor<unsigned long>>();
                break;
            case 10:
                tensor = std::make_shared<Tensor<long>>();
                break;
            case 11:
                tensor = std::make_shared<Tensor<double>>();
                break;
            case 12:
                tensor = std::make_shared<Tensor<long double>>();
                break;
            case 13:
                tensor = std::make_shared<Tensor<long long>>();
                break;
            default:
                throw std::runtime_error(
                    "ShivaMessage receiveTensor error, not implemented dtype " +
                    std::to_string(th.dtype));
            }

            tensor->header = th;
            tensor->shape = shape;
            tensor->receiveData(sock);

            return tensor;
        }

        void receiveMetadata(int sock, int metadata_size)
        {
            if (metadata_size == 0)
                return;

            std::shared_ptr<uint8_t> response_array(new uint8_t[metadata_size],
                                                    std::default_delete<uint8_t[]>());

            uint8_t *received_data = response_array.get();
            int received_size = 0;
            while (received_size < metadata_size)
            {
                int remains = metadata_size - received_size;
                int chunk_size =
                    recv(sock, &(received_data)[received_size], remains, 0);
                received_size += chunk_size;
            }

            std::string response_string((char *)response_array.get(), metadata_size);
            this->metadata = nlohmann::json::parse(response_string);
        }

        void receiveNamespace(int sock, int trail_size)
        {
            if (trail_size == 0)
                return;

            std::shared_ptr<uint8_t> response_array(new uint8_t[trail_size],
                                                    std::default_delete<uint8_t[]>());

            uint8_t *received_data = response_array.get();
            int received_size = 0;
            while (received_size < trail_size)
            {
                int remains = trail_size - received_size;
                int chunk_size =
                    recv(sock, &(received_data)[received_size], remains, 0);
                received_size += chunk_size;
            }

            std::string response_string((char *)response_array.get(), trail_size);
            this->namespace_ = response_string;
        }

        void sendHeader(int sock)
        {
            MessageHeader header = this->buildHeader();
            ssize_t size = (ssize_t)sizeof(MessageHeader);
            if (send(sock, &header, size, 0) != size)
                throw std::runtime_error("ShivaMessage sendHeader failure");
        }

        void sendMetadata(int sock)
        {
            ssize_t size = (ssize_t)this->metadata.dump().size();
            if (send(sock, this->metadata.dump().c_str(), size, 0) != size)
                throw std::runtime_error("ShivaMessage sendMetadata failure");
        }

        void sendNamespace(int sock)
        {
            ssize_t size = (ssize_t)this->namespace_.size();
            if (send(sock, this->namespace_.c_str(), size, 0) != size)
                throw std::runtime_error("ShivaMessage sendNamespace failure");
        }
    };
}

#endif // SHIVA_MESSAGE_HPP
