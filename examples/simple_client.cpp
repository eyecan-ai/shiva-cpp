#include <chrono>
#include <iostream>

#include "shiva_client.hpp"

template <typename DataType>
std::shared_ptr<shiva::Tensor<DataType>> createTensor(std::vector<uint32_t> shape,
                                                      int fill_value = 0)
{
    int total_size = 1;
    std::for_each(shape.begin(), shape.end(), [&](int n) { total_size *= n; });
    DataType *data = new DataType[total_size];

    std::shared_ptr<shiva::Tensor<DataType>> tensor =
        std::make_shared<shiva::Tensor<DataType>>();

    std::fill_n(data, total_size, fill_value);

    tensor->data = std::vector<DataType>(data, data + total_size);
    tensor->shape = shape;
    return tensor;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <server_ip> <server_port>" << std::endl;
        exit(1);
    }

    // create Shiva Client
    shiva::ShivaClient client(argv[1], atoi(argv[2]));

    // create 3 random tensors
    shiva::Tensor<uint8_t>::Ptr tensor_1 = createTensor<uint8_t>({1920, 1080, 3});
    shiva::Tensor<uint8_t>::Ptr tensor_2 = createTensor<uint8_t>({1920, 1080, 3});
    shiva::Tensor<uint32_t>::Ptr tensor_3 = createTensor<uint32_t>({10, 10});

    shiva::ShivaMessage message;

    // populate metadata
    message.metadata = {{"counter", 0},
                        {"__tensors__",
                         {
                             "tensor_1",
                             "tensor_2",
                             "tensor_3",
                         }}};

    // set command (aka namespace)
    message.command = "inference";

    // add tensors to message
    message.tensors.push_back(tensor_1);
    message.tensors.push_back(tensor_2);
    message.tensors.push_back(tensor_3);

    while (true)
    {
        auto start = std::chrono::high_resolution_clock::now();

        shiva::ShivaMessage returnMessage = client.sendAndReceiveMessage(message);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "FPS: " << 1000000 / duration.count() << std::endl;
        std::cout << "metadata: " << returnMessage.metadata.dump() << std::endl;

        message = returnMessage;
    }
}
