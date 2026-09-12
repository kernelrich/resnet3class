from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import kagglehub

"""

/home/icode/.cache/kagglehub/datasets/hoturam/bear-dataset/versions/1
/home/icode/.cache/kagglehub/datasets/hoturam/bear-dataset/versions/1/data/black/black1.jpg
path = kagglehub.dataset_download("hoturam/bear-dataset", path= "data/black/black1.jpg")

"""

def split_dataset(source_dir, train_dir, val_dir, test_size=0.2):
    """Split bear dataset into train/val"""
    source = Path(source_dir)
    
    for class_name in ["black", "grizzly", "panda", "polar", "teddy"]:
        class_path = source / "data" / class_name
        images = list(class_path.glob("*.jpg"))  # adjust extension as needed
        
        train_imgs, val_imgs = train_test_split(
            images, 
            test_size=test_size,  # 0.2 = 80/20 split
            random_state=42
        )
        
        # Create directories and copy files
        (Path(train_dir) / class_name).mkdir(parents=True, exist_ok=True)
        (Path(val_dir) / class_name).mkdir(parents=True, exist_ok=True)
        
        for img in train_imgs:
            shutil.copy(img, Path(train_dir) / class_name / img.name)
        for img in val_imgs:
            shutil.copy(img, Path(val_dir) / class_name / img.name)

if __name__ == "__main__":
	k_path = kagglehub.dataset_download("hoturam/bear-dataset")
	split_dataset(
	    source_dir=k_path,
	    train_dir="data_dir/train",
	    val_dir="data_dir/val",
	    test_size=0.2  # 80/20 split
	)

