trainingimages = torch.Tensor(file_reader.numberReader("trainingimages"))
    traininglabels = torch.LongTensor(file_reader.number_labelReader("traininglabels"))
    training_dataset = TensorDataset(trainingimages, traininglabels)

    testimages = torch.Tensor(file_reader.numberReader("testimages"))
    testlabels = torch.LongTensor(file_reader.number_labelReader("testlabels"))
    test_dataset = TensorDataset(testimages, testlabels)