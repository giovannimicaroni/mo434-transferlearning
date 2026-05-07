from pet_dataset import PetDataset

from torch.utils.data import DataLoader

from tqdm import tqdm


def main():
    dataset = PetDataset(
        root_path="/home/gimicaroni/Documents/Datasets/Oxford-Pet-Dataset/images/images"
    )
    dataset_loader = DataLoader(dataset)
    for data in tqdm(dataset_loader):
        image, label = data
        # print(image.shape, label)


if __name__ == "__main__":
    main()
