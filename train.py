import train_

if __name__ == "__main__":
    print("training autoencoder")
    train_.autoencoder.main()
    print("training classifier")
    train_.classifier.main()
    print("training finished, results can be found in `artifacts` directory")
