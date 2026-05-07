from torchvision import models


def main():
    # Load VGG16 with pre-trained ImageNet weights
    model = models.vgg16(weights="DEFAULT")
    print(model)
    model.eval()


if __name__ == "__main__":
    main()
